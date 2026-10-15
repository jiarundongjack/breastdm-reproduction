"""Summarize only all three completed, identical-protocol final Exp-2 runs."""
import csv
from pathlib import Path
import argparse
import numpy as np
from exp2_provenance import SEEDS, BASE, run_path, read_json, verify_completed, validate_results

METRICS = ("Accuracy", "Sensitivity", "Specificity", "Precision_Malignant", "Precision_Weighted", "AUC")
PAPER = {"Accuracy":.8393, "Sensitivity":.8750, "Specificity":.7500, "Precision":.8427, "AUC":.8826}


def summarize(output):
    results, reference = [], None
    for seed in SEEDS:
        run = run_path(seed)
        verify_completed(run)
        config = read_json(run / "run_config.json")
        if reference is None: reference = config["invariants"]
        if config["invariants"] != reference: raise RuntimeError("Cross-seed protocol mismatch")
        row = validate_results(run)
        assert row["seed"] == seed
        results.append(row)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    def save(name, rows):
        with (output / name).open("x", encoding="utf-8-sig", newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save("exp2_multiseed_summary.csv",results)
    stats = [{"Metric":m, "Mean":float(np.mean([r[m] for r in results])),
              "SD":float(np.std([r[m] for r in results],ddof=1)), "ddof":1, "Runs":3,
              "Definition":"author-released-code-compatible AUC" if m=="AUC" else m} for m in METRICS]
    save("exp2_multiseed_mean_sd.csv",stats)
    comparisons=[]
    for stat in stats:
        m=stat["Metric"]
        paper_key="Precision" if m.startswith("Precision_") else m
        comparisons.append({"Metric":m,"Original Table 5":PAPER[paper_key],
            **{f"seed{r['seed']}":r[m] for r in results},
            "Mean":stat["Mean"],"SD":stat["SD"],"Mean +/- SD":f"{stat['Mean']:.8f} +/- {stat['SD']:.8f}",
            "Notes":"Table 5 Precision is compared against both definitions without assigning its implementation"
              if m.startswith("Precision_") else stat["Definition"],"Units":"fraction (0-1)"})
    save("exp2_table5_comparison.csv",comparisons)
    print(output)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=BASE/"results"/"exp2_final_multiseed_summary")
    summarize(p.parse_args().output_dir)
