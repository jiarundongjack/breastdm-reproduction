from release_paths import REPO, RESULT, RUNS, OUTPUT, PAPER, PRETRAINED, EXP1_DATA, EXP2_DATA, SEG2_DATA, SEG3_DATA, RUN3, RUNTIME
import os
import sys
import importlib.util

ROOT = (REPO)
RUN_DIR = os.path.join(
    RESULT,
    "fusion_exp2_100ep_seed59317_final_multiseed"
)
SNAPSHOT = os.path.join(RUN_DIR, "code_snapshot")
DATA_ROOT = (EXP2_DATA)

loader_file = os.path.join(SNAPSHOT, "data_loader_exp2.py")

sys.path.insert(0, SNAPSHOT)
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location(
    "formal_data_loader_exp2",
    loader_file
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

Dataset = module.BreastDMNpyDataset

ds = Dataset(
    root=DATA_ROOT,
    split="test",
    augment=False
)

idx = 96

print("Dataset length:", len(ds))
print("Index:", idx)
print("Label:", ds.targets[idx])

print("\nDataset attributes:")
for name in ["samples", "files", "paths", "data", "imgs"]:
    if hasattr(ds, name):
        obj = getattr(ds, name)
        print(f"\n{name}:")
        try:
            print(obj[idx])
        except Exception:
            print(obj)

print("\nRaw object dict keys:")
print(ds.__dict__.keys())

for k, v in ds.__dict__.items():
    try:
        if hasattr(v, "__len__") and len(v) == len(ds):
            print(f"\nCandidate attribute: {k}")
            print(v[idx])
    except Exception:
        pass
