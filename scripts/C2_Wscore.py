"""Fit a healthy-reference model and calculate patient W-scores.

W = (predicted GMV - observed GMV) / healthy-reference residual SD.
Positive scores therefore indicate lower observed GMV than predicted.
See ../README.md for input formats and examples.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MAIN_DIR = Path(r" ") # change this path when running on another computer.
DATA_DIR = MAIN_DIR / "output" / "combat"
OUTPUT_DIR = MAIN_DIR / "output" / "wscore"
DEMO_FILE = DATA_DIR / "demoData.csv"
IMAGE_FILE = DATA_DIR / "GrayVol_combat.csv"
ETIV_FILE = DATA_DIR / "eTIV.csv"

DEFAULT_COVARIATES = ["Sex_Code", "Age", "PTHAND", "PTEDUCAT", "eTIV"]
ID = "Subject ID"


def read_table(path):
    """Read a ComBat output CSV with a header and string subject IDs."""
    table = pd.read_csv(path, dtype=str)
    table[ID] = table[ID].str.strip()
    return table


def align(table, ids):
    return table.set_index(ID).loc[ids].reset_index()


def numeric(table, columns):
    return table.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)


def calculate_wscores(x, y, healthy):
    design = np.column_stack([np.ones(len(x)), x])
    reference_x, reference_y = design[healthy], y[healthy]
    n, p = reference_x.shape

    beta, _, rank, _ = np.linalg.lstsq(reference_x, reference_y, rcond=None)

    residual = reference_y - reference_x @ beta
    sigma = np.sqrt(np.sum(residual ** 2, axis=0) / (n - p))

    scores = (design[~healthy] @ beta - y[~healthy]) / sigma
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covariates", nargs="+", default=DEFAULT_COVARIATES)
    parser.add_argument("--healthy-code", type=float, default=0, help="Group_Code value for healthy controls")
    parser.add_argument("--threshold", type=float, default=0, help="Scores below this value are replaced by zero")
    args = parser.parse_args(argv)

    demo = read_table(DEMO_FILE)
    image = align(read_table(IMAGE_FILE), demo[ID])
    eTIV = align(read_table(ETIV_FILE), demo[ID])
    demo["eTIV"] = numeric(eTIV, ["eTIV"]).ravel()
    roi_columns = list(image.columns[1:])
    x = numeric(demo, args.covariates)
    y = numeric(image, roi_columns)
    groups = numeric(demo, ["Group_Code"]).ravel()
    healthy = groups == args.healthy_code
    scores = calculate_wscores(x, y, healthy)
    thresholded = np.where(scores < args.threshold, 0, scores)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patient_ids = demo.loc[~healthy, ID].to_numpy()
    for filename, values in [(f"W_scores_threshold{args.threshold:g}.csv", thresholded)]:
        table = pd.DataFrame(values, columns=roi_columns)
        table.insert(0, ID, patient_ids)
        table.to_csv(OUTPUT_DIR / filename, index=False)
    demo.loc[~healthy].to_csv(OUTPUT_DIR / "W_scores_demo.csv", index=False, encoding="utf-8-sig")

    print(f"Completed: {healthy.sum()} healthy, {(~healthy).sum()} non-healthy, {len(roi_columns)} ROIs")



if __name__ == "__main__":
    main()
