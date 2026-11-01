"""Results table = CV of every run + Public / Private LB of every submitted file.

Inputs:
  results/cv_results.csv  CV of every run (taken from the notebook outputs; or pass --registry with the
                          experiments/registry.csv that the notebook writes, which also has GPU-hours / peak memory)
  results/lb_scores.csv   one row per submitted file: run, submission_file, lb_public, lb_private
Output:
  results/results_table.csv and results/results_table.md
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", help="experiments/registry.csv from the Kaggle output (optional)")
    args = ap.parse_args()

    cv = pd.read_csv(ROOT / "cv_results.csv")
    if args.registry:
        reg = (pd.read_csv(args.registry).sort_values("finished_at").drop_duplicates("name", keep="last")
               [["name", "gpu_hours", "peak_mem_gb", "gpu"]].rename(columns={"name": "run"}))
        cv = cv.drop(columns=[c for c in ("gpu_hours", "peak_mem_gb") if c in cv]).merge(reg, on="run", how="left")
    lb = pd.read_csv(ROOT / "lb_scores.csv")
    tab = cv.merge(lb.groupby("run")[["lb_public", "lb_private"]].first().reset_index(), on="run", how="left")
    tab["cv"] = tab["cv_mean"].map("{:.2f}".format) + " ± " + tab["cv_std"].map("{:.2f}".format)
    tab.to_csv(ROOT / "results_table.csv", index=False)
    cols = [c for c in ("block", "run", "family", "folds", "cv", "cv_median", "emd_fine", "emd_sand", "emd_gravel",
                        "lb_public", "lb_private", "gpu_hours", "peak_mem_gb") if c in tab]
    (ROOT / "results_table.md").write_text(tab[cols].to_markdown(index=False, floatfmt=".2f"))
    print(f"{len(tab)} runs, {tab['lb_public'].notna().sum()} with a public LB score")


if __name__ == "__main__":
    main()
