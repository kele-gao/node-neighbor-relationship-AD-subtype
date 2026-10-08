"""Harmonize ROI GMV using neuroCombat, preserving study covariates.

Singleton sites retain their original measurements, matching the study script.
See ../README.md for inputs, assumptions, and downstream usage.
"""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


MAIN_DIR = Path(r" ") # change this path when running on another computer.
DATA_DIR = MAIN_DIR / "data"
OUTPUT_DIR = MAIN_DIR / "output" / "combat"

DEMO_FILE = DATA_DIR / "demographics.csv"
IMAGE_FILE = DATA_DIR / "GrayVol.csv"
ETIV_FILE = DATA_DIR / "eTIV.csv"

ID = "Subject ID" 


def read_table(path, id_column=None):
    table = pd.read_csv(path, dtype=str)
    column = id_column or table.columns[0]
    ids = table[column].str.strip()
    table = table.drop(columns=[column])
    table.insert(0, ID, ids)
    return table


def align(table, ids):
    # Match original behavior: demographics selects the analysis cohort.
    return table.set_index(ID).loc[ids].reset_index()


def numbers(table, columns, label):
    result = table[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: numeric values must be finite and complete")
    return result


def harmonize(features, covars, batch_col, categorical_cols, continuous_cols):
    """Only sites with >=2 subjects enter the ComBat fit."""
    counts = covars[batch_col].value_counts()
    multi = counts[counts > 1].index
    mask = covars[batch_col].isin(multi).to_numpy()
    corrected = features.astype(float, copy=True)
    performed = len(multi) >= 2

    from neuroCombat import neuroCombat
    result = neuroCombat(
        dat=features[:, mask], covars=covars.loc[mask].reset_index(drop=True),
        batch_col=batch_col, categorical_cols=categorical_cols,
        continuous_cols=continuous_cols, eb=True, parametric=True,
        mean_only=False, ref_batch=None,
    )
    values = np.asarray(result["data"], dtype=float)
    corrected[:, mask] = values
    status = np.where(mask & performed, "harmonized", "unchanged")
    return corrected, status, performed, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-col", default="SITEID")
    parser.add_argument("--categorical-cols", nargs="*", default=["Sex", "PTHAND", "Group_Code"])
    parser.add_argument("--continuous-cols", nargs="*", default=["Age", "PTEDUCAT", "eTIV"])
    args = parser.parse_args(argv)
    output_dir = OUTPUT_DIR
    columns = [args.batch_col] + args.categorical_cols + args.continuous_cols
    demo = read_table(DEMO_FILE, ID)
    image = align(read_table(IMAGE_FILE, ID), demo[ID])
    etiv = align(read_table(ETIV_FILE, ID), demo[ID])

    model_demo = demo.copy()
    model_demo["eTIV"] = etiv["eTIV"].to_numpy()
    covars = model_demo[columns].copy()
    for column in [args.batch_col] + args.categorical_cols:
        covars[column] = covars[column].str.strip()
        if covars[column].isna().any() or covars[column].eq("").any():
            raise ValueError(f"Missing batch/category values in {column}")
    if args.continuous_cols:
        covars[args.continuous_cols] = numbers(model_demo, args.continuous_cols, "Continuous covariates")
    roi_columns = list(image.columns[1:])
    if not roi_columns:
        raise ValueError("GMV input must contain ROI columns")
    image[roi_columns] = numbers(image, roi_columns, "GMV")
    corrected, status, performed, counts = harmonize(
        image[roi_columns].to_numpy(dtype=float).T, covars, args.batch_col,
        args.categorical_cols, args.continuous_cols)
    output = image.copy()
    output[roi_columns] = corrected.T

    output_dir.mkdir(parents=True, exist_ok=True)

    demo.to_csv(output_dir / "demoData.csv", index=False, encoding="utf-8-sig")
    etiv.to_csv(output_dir / "eTIV.csv", index=False)
    output.to_csv(output_dir / "GrayVol_combat.csv", index=False)


    print(f"Completed: {len(demo)} subjects; ComBat performed: {performed}")



if __name__ == "__main__":
    main()
