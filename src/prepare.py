"""
Step 1 - Prepare the Lending Club data.

  * keep only loans with a resolved outcome (Fully Paid / Charged Off / Default)
  * define the binary target `default`
  * DROP every column that is only known after origination (leakage)
  * engineer a few origination-time features
  * create a time-based train/test flag (train on older vintages, test on newer)

Usage:  python src/prepare.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def find_raw() -> Path:
    for p in config.RAW_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No raw data found in data/. Download the Lending Club file from Kaggle "
        "(see README) or run `python src/make_sample_data.py` for a synthetic smoke test."
    )


def parse_month(s: pd.Series) -> pd.Series:
    """Lending Club dates look like 'Dec-2015'."""
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def emp_length_to_years(s: pd.Series) -> pd.Series:
    return (s.astype(str)
             .str.replace("10+ years", "10", regex=False)
             .str.replace("< 1 year", "0", regex=False)
             .str.extract(r"(\d+)")[0]
             .astype(float))


def main() -> None:
    raw = find_raw()
    print(f"reading {raw.name} ...")
    df = pd.read_csv(raw, low_memory=False)
    print(f"  raw shape: {df.shape}")

    # ---- target -----------------------------------------------------------
    df = df[df["loan_status"].isin(config.GOOD_STATUSES + config.BAD_STATUSES)].copy()
    df[config.TARGET] = df["loan_status"].isin(config.BAD_STATUSES).astype(int)
    print(f"  resolved loans: {len(df):,}   default rate: {df[config.TARGET].mean():.2%}")

    # ---- leakage & identifier removal (the single most important step) -----
    to_drop = [c for c in config.LEAKAGE_COLS + config.ID_TEXT_COLS if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"  dropped {len(to_drop)} leakage/id columns: {', '.join(sorted(to_drop))}")

    # ---- feature engineering (origination-time only) -----------------------
    df["issue_dt"] = parse_month(df["issue_d"])
    df["issue_year"] = df["issue_dt"].dt.year
    if "earliest_cr_line" in df.columns:
        ecl = parse_month(df["earliest_cr_line"])
        df["credit_history_years"] = (df["issue_dt"] - ecl).dt.days / 365.25
    if "emp_length" in df.columns:
        df["emp_length_years"] = emp_length_to_years(df["emp_length"])
    if "int_rate" in df.columns and df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)
    if "revol_util" in df.columns and df["revol_util"].dtype == object:
        df["revol_util"] = df["revol_util"].str.rstrip("%").astype(float)
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.strip()
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)

    # ---- time-based split --------------------------------------------------
    df["is_test"] = (df["issue_year"] >= config.TEST_FROM_YEAR).astype(int)
    n_tr, n_te = (df["is_test"] == 0).sum(), (df["is_test"] == 1).sum()
    print(f"  train (< {config.TEST_FROM_YEAR}): {n_tr:,}   test (>= {config.TEST_FROM_YEAR}): {n_te:,}")
    if n_tr == 0 or n_te == 0:
        raise ValueError("Empty train or test split - adjust TEST_FROM_YEAR in config.py")

    # ---- keep only modelling columns --------------------------------------
    num = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]
    keep = num + cat + [config.TARGET, "is_test", "issue_year"]
    df = df[keep]
    for c in cat:
        df[c] = df[c].astype("string").fillna("MISSING")

    df.to_pickle(config.PREPARED)
    print(f"  saved {config.PREPARED}  ({len(num)} numeric, {len(cat)} categorical features)")


if __name__ == "__main__":
    main()
