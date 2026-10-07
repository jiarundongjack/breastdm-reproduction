"""Run the paper task folders through an isolated, conventional Python layout.

Only standard-library modules are needed to list/stage tasks. Source files in
task folders remain canonical; temporary copies are removed after execution.
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
PACKAGES = {
    'shared_2d_3d_model_imports': 'src/__init__.py',
    'shared_2d_3d_training_imports': 'train_utils/__init__.py',
    'shared_2d_3d_metrics': 'train_utils/distributed_utils.py',
    'shared_2d_3d_dice_loss': 'train_utils/dice_coefficient_loss.py',
    'shared_2d_3d_training': 'train_utils/train_and_eval.py',
    'table7_metrics': 'train_utils/segmentation3d_metrics.py',
}
MODULES = {'classification_metrics', 'exp1_data_loader', 'exp2_data_loader',
           'exp1_model', 'exp2_model', 'exp1_run_records', 'exp2_run_records',
           'shared_transformer', 'data_loader', 'augmentation', 'unet_model',
           *PACKAGES}

def sources():
    return {str(p.parent.relative_to(CODE)).replace('\\', '/'): p
            for category in ['classification', 'segmentation_2d', 'segmentation_3d']
            for p in (CODE/category).glob('*/*.py')}

def stage(destination, classification):
    for task, source in sources().items():
        if task.startswith('classification/') != classification:
            continue
        subfolder = source.parent.name
        name = PACKAGES.get(subfolder, source.name)
        if subfolder == 'unet_model': name = 'src/' + source.name
        target = destination/name
        if target.exists(): raise RuntimeError(f'Duplicate runtime module: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(CODE/'release_paths.py', destination/'release_paths.py')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', nargs='?', help='category/task_folder; use --list')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--output-root', type=Path, default=ROOT/'result/reproduced',
                        help='destination for analysis outputs, separate from formal runs')
    parser.add_argument('--runs-root', type=Path,
                        help='read a newly trained three-seed run set instead of formal result/')
    args, extra = parser.parse_known_args()
    tasks = sources()
    if args.list or args.task is None:
        for task, source in sorted(tasks.items()):
            if source.parent.name not in MODULES: print(task)
        return 0
    if args.task not in tasks: parser.error('Unknown task. Use --list.')
    if extra and extra[0] == '--': extra = extra[1:]
    output = args.output_root.resolve()
    formal = [p.resolve() for p in (ROOT/'result').iterdir()
              if p.is_dir() and (p.name.startswith('fusion_') or p.name.startswith('seg'))]
    def protect(path):
        path = path.resolve()
        if any(path == p or p in path.parents for p in formal):
            parser.error('Choose a new output location outside the eight formal runs.')
    protect(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = args.runs_root.resolve() if args.runs_root else ROOT/'result'
    training = args.task.endswith(('exp1_training', 'exp2_training',
                                   'table6_fig6_7_training', 'table7_fig8_9_12_training'))
    if '--output-dir' in extra:
        i = extra.index('--output-dir') + 1
        if i == len(extra): parser.error('--output-dir needs a path')
        target = Path(extra[i]).resolve()
        protect(target)
        extra[i] = str(target)
        if training: runs = target.parent
    if '--path' in extra:
        i=extra.index('--path')+1
        extra[i]=str(Path(extra[i]).resolve())
    for flag in ['--data-root', '--data-path', '--resume', '--test_images', '--ground_truth_images', '-i', '-g']:
        if flag in extra:
            i=extra.index(flag)+1
            extra[i]=str(Path(extra[i]).resolve())
    env = os.environ.copy()
    env.update(BREASTDM_ROOT=str(ROOT), BREASTDM_OUTPUT=str(output),
               BREASTDM_RUNS=str(runs), PYTHONDONTWRITEBYTECODE='1',
               PYTHONIOENCODING='utf-8', PYTHONUTF8='1', MPLBACKEND='Agg')
    with tempfile.TemporaryDirectory(prefix='breastdm_runtime_') as tmp:
        runtime=Path(tmp)
        stage(runtime, args.task.startswith('classification/'))
        return subprocess.call([sys.executable, '-B', str(runtime/tasks[args.task].name), *extra],
                               cwd=runtime, env=env)

if __name__ == '__main__':
    raise SystemExit(main())
