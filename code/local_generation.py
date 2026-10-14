"""Create a private workspace and call existing inference/manifest functions.

No training entry point is called. See LOCAL_GENERATION.md before execution.
Only `init` operates from the public checkout; generation requires its private copy.
"""
import argparse
import ast
import csv
import importlib
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MARKER = '.breastdm-local-workspace'
SEG_RUNS = {
    'seg2d': 'seg2d_unet_seed123_gpu_20260911',
    'seg3d': 'seg3d_unet_locked_volume_ppv_seed42_20260911',
}
SEEDS = {'exp1': (31415, 27182, 16180), 'exp2': (48271, 59317, 84629)}


def initialize():
    """Copy tracked working files only; never copy Git state, data or weights."""
    if (ROOT / MARKER).exists():
        raise RuntimeError('Run init from the public checkout, not its private copy.')
    destination = (ROOT / '.local/reproduction').resolve()
    if not destination.is_relative_to(ROOT) or destination.exists():
        raise FileExistsError('The private destination must be new: ' + str(destination))
    names = subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', '-z'], text=True).split('\0')
    files = []
    for name in filter(None, names):
        source = ROOT / name
        if not source.exists():  # also usable before the cleanup commit
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
            raise ValueError('Refusing linked source: ' + name)
        if name.startswith(('data/', '.local/')) or source.suffix.lower() in {
            '.pt', '.pth', '.ckpt', '.pkl', '.npy', '.npz', '.dcm', '.nii', '.gz'}:
            continue
        files.append((name, source))
    if not any(name == 'code/local_generation.py' for name, _ in files):
        raise RuntimeError('Use a checkout with local_generation.py staged or committed.')
    destination.mkdir(parents=True)
    for name, source in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (destination / MARKER).write_text('Private local outputs; do not publish.\n', encoding='utf-8')
    print(destination)


def evaluator_namespace(source):
    """Load unchanged imports, metric-field constants and the existing test function.

    Classification training modules have top-level argument parsing and dataset
    construction. Do not import those modules or execute their training setup.
    """
    tree = ast.parse(source.read_text(encoding='utf-8-sig'), filename=str(source))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == 'evaluate_test_set':
            selected.append(node)
        elif isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) and t.id in {'TEST_METRIC_FIELDS', 'TEST_RESULT_FIELDS'}
                for t in node.targets):
            selected.append(node)
    namespace = {'__file__': str(source), '__name__': '_local_evaluator'}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    if 'evaluate_test_set' not in namespace:
        raise RuntimeError('Existing test evaluator not found.')
    return namespace


def classification(args, runtime):
    import torch
    import numpy as np
    exp = args.experiment
    if args.seed not in SEEDS[exp]:
        raise ValueError(f'Use an existing {exp} seed: {SEEDS[exp]}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    paths = importlib.import_module('release_paths')
    data_root = paths.EXP1_DATA if exp == 'exp1' else paths.EXP2_DATA
    loader_module = importlib.import_module('data_loader' if exp == 'exp1' else 'data_loader_exp2')
    run = ROOT / 'result' / f'fusion_{exp}_100ep_seed{args.seed}_final_multiseed'
    run.mkdir(parents=True, exist_ok=True)
    if args.action == 'manifest':
        target = run / 'dataset_manifest.csv'
        if target.exists():
            raise FileExistsError(target)
        datasets = [loader_module.BreastDMNpyDataset(str(data_root), split=s, augment=False)
                    for s in ('train', 'val', 'test')]
        provenance = importlib.import_module(exp + '_provenance')
        _, manifest = provenance.collect_evidence(
            argparse.Namespace(path=str(data_root), gpu=args.device), datasets)
        with target.open('xb') as stream:
            stream.write(manifest)
        return
    target = run / 'test_predictions_two_class.csv'
    if target.exists():
        raise FileExistsError(target)
    dataset = loader_module.BreastDMNpyDataset(str(data_root), split='test', augment=False)
    device = torch.device(args.device)
    model_module = importlib.import_module('fusionModels' if exp == 'exp1' else 'fusionModels_exp2')
    model = model_module.FusionM(num_classes=2, load_vit=False, load_se=False).to(device)
    model.load_state_dict(torch.load(run / 'best_fusion_model.pth', map_location=device, weights_only=True))
    namespace = evaluator_namespace(runtime / ('train.py' if exp == 'exp1' else 'train_exp2.py'))
    namespace.update(seed=args.seed, target_test_loader=torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda'))
    namespace['evaluate_test_set'](model, str(run))


def segmentation(args):
    import torch
    import numpy as np
    three_d = args.experiment == 'seg3d'
    seed = 42 if three_d else 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    module = importlib.import_module('train3d' if three_d else 'train')
    paths = importlib.import_module('release_paths')
    run = ROOT / 'result' / SEG_RUNS[args.experiment]
    if (run / 'test_predictions.csv').exists():
        raise FileExistsError(run / 'test_predictions.csv')
    dataset = module.DriveDataset(
        str(paths.SEG3_DATA if three_d else paths.SEG2_DATA), split='test',
        transforms=module.get_transform(train=False, input_size=224),
        **({'input_size': 224, 'depth': 8} if three_d else {}))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                       num_workers=0, pin_memory=True)
    device = torch.device(args.device)
    model = module.create_model(aux=False, num_classes=2).to(device)
    # Same checkpoint loading convention as the existing segmentation entry points.
    checkpoint = torch.load(run / 'best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    _, dice, miou, ppv, details = module.evaluate(
        model, loader, device=device, num_classes=2,
        **({} if three_d else {'return_ppv_details': True, 'header': 'Test:'}))
    with (run / 'test_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['best_epoch', 'test_dice', 'test_miou', 'test_ppv', *details.keys()])
        writer.writerow([checkpoint['epoch'] + 1, float(dice), float(miou), float(ppv), *details.values()])
    records = module.save_test_predictions(model, loader, dataset, device, str(run))
    module.save_representative_examples(model, dataset, records, device, str(run))


def legacy_selection():
    """Supply only the local path to an existing legacy selection script."""
    run = ROOT / 'result' / SEG_RUNS['seg3d']
    source = run / 'find_overseg_case.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'), filename=str(source))
    matches = [n for n in tree.body if isinstance(n, ast.Assign)
               and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id == 'run']
    if len(matches) != 1:
        raise RuntimeError('Expected one legacy run-path assignment.')
    matches[0].value = ast.Constant(str(run))
    ast.fix_missing_locations(tree)
    os.chdir(run)
    exec(compile(tree, str(source), 'exec'), {'__name__': '__main__', '__file__': str(source)})
    runpy.run_path(str(run / 'make_top3_overseg_figures.py'), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'manifest', 'predict', 'legacy-selection'))
    parser.add_argument('--experiment', choices=('exp1', 'exp2', 'seg2d', 'seg3d'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.action == 'init':
        initialize()
        return
    if not (ROOT / MARKER).is_file() or ROOT.name != 'reproduction' or ROOT.parent.name != '.local':
        parser.error('Use .local/reproduction/code/local_generation.py after init.')
    if args.action == 'legacy-selection':
        legacy_selection()
        return
    if args.experiment is None:
        parser.error('--experiment is required.')
    is_classification = args.experiment.startswith('exp')
    if args.action == 'manifest' and not is_classification:
        parser.error('manifest is for classification; segmentation predictions contain paths.')
    if is_classification and args.seed is None:
        parser.error('--seed is required for classification.')
    os.environ.update(BREASTDM_ROOT=str(ROOT), BREASTDM_RUNS=str(ROOT / 'result'),
                      BREASTDM_OUTPUT=str(ROOT / 'result/reproduced'), MPLBACKEND='Agg')
    sys.dont_write_bytecode = True
    from run import stage
    with tempfile.TemporaryDirectory(prefix='breastdm_local_') as temporary:
        runtime = Path(temporary)
        stage(runtime, is_classification)
        sys.path.insert(0, str(runtime))
        if is_classification:
            classification(args, runtime)
        else:
            segmentation(args)


if __name__ == '__main__':
    main()
