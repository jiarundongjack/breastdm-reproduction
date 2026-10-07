"""Evidence and fail-closed checks for the unified Exp-2 protocol; no training."""
from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SEEDS = (48271, 59317, 84629)
BASE = Path(__file__).resolve().parent
CODE_FILES = ("release_paths.py", "train_exp2.py", "classification_auc.py", "data_loader_exp2.py", "fusionModels_exp2.py",
              "VIT_model.py", "exp2_provenance.py",
              "summarize_exp2_final_multiseed.py")
AUC_DEFINITION = "author-released-code-compatible AUC: full-set two-class softmax and one-hot labels flattened -> roc_curve -> auc"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def run_path(seed):
    return RUNS / f"fusion_exp2_100ep_seed{seed}_final_multiseed"


def validate_output(path):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing nonempty output directory: {path}")
    return path


def validate_args(args):
    if args.model != "fusion" or args.num_class != 2 or args.batch_size != 32 or args.auto_split != "0":
        raise ValueError("Final Exp-2 requires model=fusion, num_class=2, batch-size=32, auto_split=0")
    if args.random_seed not in SEEDS:
        raise ValueError(f"Final Exp-2 seeds are {SEEDS}")
    if Path(args.path).resolve() != Path((EXP2_DATA)).resolve():
        raise ValueError("Final Exp-2 uses the existing img17Se")
    output = validate_output(args.output_dir)
    if output != run_path(args.random_seed).resolve():
        raise ValueError(f"Expected isolated final output: {run_path(args.random_seed)}")
    return output


def collect_evidence(args, datasets):
    """Read/hash inputs without consuming any training RNG or evaluating the model."""
    rows, counts, patient_sets = [], {}, {}
    for split, ds in zip(("train", "val", "test"), datasets):
        patients = set()
        for sample, label in ds.samples:
            sample = Path(sample).resolve()
            relative = sample.relative_to(Path(args.path).resolve())
            patient = relative.parts[2]
            patients.add(patient)
            array = np.load(sample, mmap_mode="r", allow_pickle=False)
            if array.ndim != 3 or array.shape[-1] != 17 or array.dtype != np.uint8:
                raise ValueError(f"Not HWD with 17 uint8 channels: {sample}")
            rows.append({"split": split, "class": "Benign" if label == 0 else "Malignant",
                         "patient": patient, "relative_path": relative.as_posix(),
                         "sha256": sha256(sample), "shape": str(tuple(array.shape)), "dtype": str(array.dtype)})
        counts[split] = {"samples": len(ds), "patients": len(patients)}
        patient_sets[split] = patients
        expected_classes = {"train": (327,875), "val": (24,93), "test": (114,289)}
        actual = tuple(sum(int(label == i) for _,label in ds.samples) for i in (0,1))
        if actual != expected_classes[split]:
            raise ValueError(f"Class counts mismatch: {split} {actual}")
    expected = {"train": {"samples": 1202, "patients": 166},
                "val": {"samples": 117, "patients": 19}, "test": {"samples": 403, "patients": 47}}
    if counts != expected or any(patient_sets[a] & patient_sets[b] for a,b in
                                 (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError(f"Dataset counts or patient separation mismatch: {counts}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    manifest = buffer.getvalue().encode("utf-8")
    code = {name: sha256(BASE / name) for name in CODE_FILES}
    weights = {str(PRETRAINED / name): sha256(PRETRAINED / name) for name in
               ("jx_vit_base_patch16_224_in21k-e5005f0a.pth", "se_resnet50-ce0d4300.pth")}
    expected_weights = {
        "jx_vit_base_patch16_224_in21k-e5005f0a.pth": "e5005f0a7a836470c1dbd39d848eabf0a5b1d8f9a066aa46e6ff00261db44cb5",
        "se_resnet50-ce0d4300.pth": "ce0d430017d3f4aa6b5658c72209f3bfffb060207fd26a2ef0b203ce592eba01"}
    if {Path(k).name:v for k,v in weights.items()} != expected_weights:
        raise ValueError("Locked pretrained SHA-256 mismatch")
    environment = {"python_executable": sys.executable, "python_version": platform.python_version(),
                   "pytorch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                   "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "platform": platform.platform(),
                   "packages": {p: importlib.metadata.version(p) for p in
                                ("torchvision", "numpy", "scikit-learn", "pretrainedmodels", "pandas")},
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    protocol = dict(model="fusion", num_classes=2, dataset_root=str(Path(args.path).resolve()),
                    counts=counts, class_counts={"train":{"Benign":327,"Malignant":875},
                    "val":{"Benign":24,"Malignant":93}, "test":{"Benign":114,"Malignant":289}}, auto_split=0, batch_size=32, epochs=100, optimizer="SGD",
                    initial_lr=0.01, momentum=0.9, weight_decay=0.01, sgd_foreach=False,
                    mixed_precision_train=True, train_autocast="torch.cuda.amp.autocast(); default dtype=float16",
                    grad_scaler="torch.cuda.amp.GradScaler(); scale/backward/step/update",
                    grad_scaler_settings=dict(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True),
                    validation_autocast=False, test_autocast=False, class_weights_used=False,
                    input_dtype="uint8", channel_semantics="cannot be independently verified", patient_overlap=0,
                    lr_schedule="max(0.01 * 0.1 ** (epoch // 10), 1e-5); epoch range 1..100",
                    loss="nll_loss(log_softmax(logits, dim=1), label); mean; no class weights",
                    input_channels=17, input_size=[96,96],
                    preprocessing="HWD -> CHW -> float / 255 -> bilinear 96x96; align_corners=False",
                    augmentation="train: HFlip p=.5; scale U(.9,1.1), center crop/zero pad; retain U(.9,1), random crop then bilinear resize; val/test augment=False",
                    train_shuffle=True, eval_shuffle=False, num_workers=0, drop_last=False,
                    freeze=False, pretrained_weights=weights,
                    pretrained_adaptation="CNN and ViT w.repeat(1,6,1,1)[:, :17] * (3.0/17.0); ViT positional grid bicubic 14x14->6x6; load once, strict=False matching keys",
                    checkpoint_selection="Validation AUC strictly increases (val_auc > best_val_auc); ties keep earlier; Test excluded",
                    validation_auc=AUC_DEFINITION, test_auc=AUC_DEFINITION,
                    hard_decision="torch.argmax(logits, dim=1)",
                    precision="Precision = Precision_Malignant = TP/(TP+FP); undefined returns NaN",
                    precision_weighted='precision_score(full labels, full predictions, average="weighted", zero_division=0)',
                    gpu_argument=args.gpu, environment=environment)
    return {"protocol": protocol, "code_sha256": code,
            "dataset_manifest_sha256": hashlib.sha256(manifest).hexdigest()}, manifest


def check_predecessors(seed, evidence):
    for earlier in SEEDS[:SEEDS.index(seed)]:
        run = run_path(earlier)
        verify_completed(run)
        if read_json(run / "run_config.json")["seed"] != earlier:
            raise RuntimeError("Predecessor seed mismatch")
        if read_json(run / "run_config.json")["invariants"] != evidence:
            raise RuntimeError(f"Protocol/code/data/environment differs from completed seed {earlier}; stop")


def save_evidence(output, args, evidence, manifest):
    output = validate_output(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_config.json", dict(seed=args.random_seed, output_directory=str(output),
               timestamp=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
               invariants=evidence))
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name, expected in evidence["code_sha256"].items():
        raw = (BASE / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError(f"Code changed during snapshot: {name}")
        (snapshot / name).write_bytes(raw)
    write_json(output / "code_sha256.json", evidence["code_sha256"])
    (output / "dataset_manifest.csv").write_bytes(manifest)
    (output / "dataset_manifest_sha256.txt").write_text(evidence["dataset_manifest_sha256"] + "\n")


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file
    def write(self, text):
        self.console.write(text); self.file.write(text); self.file.flush()
    def flush(self):
        self.console.flush(); self.file.flush()


def begin_run(args, datasets):
    output = validate_args(args)
    evidence, manifest = collect_evidence(args, datasets)
    if not torch.cuda.is_available():
        raise RuntimeError("Final Exp-2 requires available CUDA for the locked AMP pipeline")
    if args.preflight_only:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="breastdm_exp2_preflight_") as temporary:
            path = Path(temporary) / "evidence"
            save_evidence(path, args, evidence, manifest)
            assert sha256(path / "dataset_manifest.csv") == evidence["dataset_manifest_sha256"]
            assert read_json(path / "code_sha256.json") == evidence["code_sha256"]
            try:
                validate_output(path)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Nonempty output protection failed")
        try:
            check_predecessors(args.random_seed, evidence)
            print("Sequence check: eligible")
        except FileNotFoundError:
            print("Sequence check: formal launch BLOCKED until predecessor completes")
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        print("PREFLIGHT PASSED: temporary evidence only; no model, training, inference or formal run directory created.")
        return None
    check_predecessors(args.random_seed, evidence)
    save_evidence(output, args, evidence, manifest)
    log = (output / "training_log.txt").open("x", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    print(json.dumps(evidence["protocol"], ensure_ascii=False, indent=2))
    print(f"Seed={args.random_seed}; output={output}; samples train/val/test=1202/117/403")
    return evidence


def validate_results(run):
    from classification_auc import classification_test_metrics
    def read(name):
        with (run / name).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    history = read("training_history.csv")
    if [int(r["Epoch"]) for r in history] != list(range(1,101)):
        raise ValueError("Incomplete 100-epoch history")
    best, best_epoch = -1., 0
    for r in history:
        value = float(r["Val AUC"])
        assert np.isfinite(value) and 0 <= value <= 1
        if value > best: best, best_epoch = value, int(r["Epoch"])
        assert int(r["Best Epoch"]) == best_epoch and float(r["Best Val AUC"]) == best
    pred = read("test_predictions_two_class.csv")
    assert len(pred) == 403
    metric = classification_test_metrics([int(r["True Label"]) for r in pred],
        [[float(r["Benign Probability"]), float(r["Malignant Probability"])] for r in pred],
        [int(r["Predicted Label"]) for r in pred], include_counts=True)
    saved_rows = read("author_style_metrics.csv")
    assert len(saved_rows) == 1
    saved = saved_rows[0]
    config = read_json(run / "run_config.json")
    assert int(saved["Seed"]) == config["seed"] and config["seed"] in SEEDS
    assert saved["Experiment"] == "Exp-2"
    reload_record = read_json(run / "best_checkpoint_reload.json")
    assert reload_record == {"Best Epoch": best_epoch, "Best Validation AUC": best,
                             "sha256": sha256(run / "best_fusion_model.pth")}
    for k,v in metric.items():
        assert np.isfinite(v) and np.isclose(float(saved[k]),v,rtol=0,atol=1e-12), k
    assert metric["N"] == sum(metric[k] for k in ("TP","TN","FP","FN")) == 403
    assert (run / "best_fusion_model.pth").is_file()
    return dict(seed=int(saved["Seed"]), **{"Best Epoch":best_epoch, "Best Validation AUC":best}, **metric)


def finish_run(args, datasets, evidence):
    current, _ = collect_evidence(args, datasets)
    if current != evidence:
        raise RuntimeError("Inputs/code/environment changed during training; do not proceed")
    run = Path(args.output_dir).resolve()
    if read_json(run / "run_config.json")["invariants"] != evidence:
        raise RuntimeError("Saved protocol differs from launch evidence")
    if read_json(run / "code_sha256.json") != evidence["code_sha256"]:
        raise RuntimeError("Saved code hash manifest differs")
    for name, h in evidence["code_sha256"].items():
        if sha256(run / "code_snapshot" / name) != h:
            raise RuntimeError(f"Snapshot changed: {name}")
    if sha256(run / "dataset_manifest.csv") != evidence["dataset_manifest_sha256"]:
        raise RuntimeError("Saved dataset manifest changed")
    result = validate_results(run)
    for name in ("training_log.txt", "code_sha256.json", "dataset_manifest.csv", "dataset_manifest_sha256.txt",
                 "accuracy_curve.png", "loss_curve.png", "test_roc_curve.png", "test_pr_curve.png"):
        if not (run / name).is_file(): raise FileNotFoundError(run / name)
    print("COMPLETED: 100 epochs, best checkpoint reloaded, full Test and integrity checks passed.")
    sys.stdout.flush(); sys.stderr.flush()
    hashes = {str(p.relative_to(run)):sha256(p) for p in run.rglob("*") if p.is_file()}
    write_json(run / "completion.json", {"status":"complete", "result":result, "files_sha256":hashes})


def verify_completed(run):
    marker = read_json(run / "completion.json")
    if marker["status"] != "complete": raise RuntimeError(f"Incomplete run: {run}")
    required = {"run_config.json", "training_history.csv", "training_log.txt", "author_style_metrics.csv",
                "test_predictions_two_class.csv", "best_fusion_model.pth", "best_checkpoint_reload.json",
                "code_sha256.json", "dataset_manifest.csv", "dataset_manifest_sha256.txt"}
    if not required <= marker["files_sha256"].keys():
        raise RuntimeError("Incomplete completion evidence")
    for name,h in marker["files_sha256"].items():
        if sha256(run / name) != h: raise RuntimeError(f"Completed run changed: {run / name}")
    validate_results(run)
