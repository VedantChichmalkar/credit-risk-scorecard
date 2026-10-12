"""
Generate a SYNTHETIC Lending-Club-shaped CSV so the pipeline can be smoke-tested
without the real data. The numbers it produces are NOT real and must never be
reported on a resume. Replace with the Kaggle file for real results (see README).

Usage:  python src/make_sample_data.py --rows 60000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main(rows: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    grade = rng.choice(list("ABCDEFG"), rows, p=[.17, .29, .28, .15, .07, .03, .01])
    grade_idx = np.array([ "ABCDEFG".index(g) for g in grade ])
    sub_grade = [f"{g}{rng.integers(1, 6)}" for g in grade]
    int_rate = np.clip(6 + grade_idx * 3.2 + rng.normal(0, 1.2, rows), 5, 31)
    term = rng.choice([" 36 months", " 60 months"], rows, p=[.72, .28])
    loan_amnt = rng.choice(np.arange(1000, 40001, 500), rows)
    annual_inc = np.clip(rng.lognormal(11.1, .55, rows), 8000, 600000)
    dti = np.clip(rng.normal(18, 8, rows), 0, 45)
    fico_low = np.clip((720 - grade_idx * 9 + rng.normal(0, 22, rows)).round(-0), 660, 845)
    emp_length = rng.choice(
        ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
         "6 years", "7 years", "8 years", "9 years", "10+ years", np.nan],
        rows, p=[.08, .07, .09, .08, .06, .06, .05, .05, .05, .04, .32, .05])
    home = rng.choice(["RENT", "MORTGAGE", "OWN", "ANY"], rows, p=[.40, .48, .11, .01])
    verif = rng.choice(["Not Verified", "Source Verified", "Verified"], rows)
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "other",
         "major_purchase", "medical", "small_business", "car", "moving", "vacation"],
        rows, p=[.57, .21, .06, .05, .02, .02, .02, .02, .02, .01])
    state = rng.choice(["CA", "TX", "NY", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI"], rows)
    years = rng.choice(range(2012, 2019), rows, p=[.04, .08, .12, .18, .18, .20, .20])
    months = rng.integers(1, 13, rows)
    issue_d = [f"{pd.Timestamp(2000, m, 1).strftime('%b')}-{y}" for m, y in zip(months, years)]
    earliest = [f"{pd.Timestamp(2000, rng.integers(1, 13), 1).strftime('%b')}-{y - rng.integers(5, 30)}"
                for y in years]
    inq = rng.poisson(.7, rows)
    delinq = rng.poisson(.3, rows)
    open_acc = rng.poisson(11, rows)
    pub_rec = rng.poisson(.2, rows)
    revol_bal = np.clip(rng.lognormal(9.3, 1.0, rows), 0, 300000)
    revol_util = np.clip(rng.normal(50, 24, rows), 0, 130)
    total_acc = open_acc + rng.poisson(13, rows)
    mort_acc = rng.poisson(1.6, rows)
    bankr = (rng.random(rows) < .12).astype(int)
    app_type = rng.choice(["Individual", "Joint App"], rows, p=[.95, .05])
    init_list = rng.choice(["w", "f"], rows, p=[.7, .3])

    # Latent default probability driven by sensible risk factors.
    logit = (-3.6 + .32 * grade_idx + .018 * (int_rate - 12) + .022 * (dti - 18)
             - .012 * (fico_low - 700) + .25 * (term == " 60 months")
             + .35 * inq.clip(0, 4) / 4 + .20 * delinq.clip(0, 3) / 3 + .006 * (revol_util - 50)
             + .30 * (home == "RENT") + .5 * (purpose == "small_business")
             - .25 * (np.array([str(e) for e in emp_length]) == "10+ years")
             + rng.normal(0, .5, rows))
    p = 1 / (1 + np.exp(-logit))
    bad = rng.random(rows) < p
    loan_status = np.where(bad, "Charged Off", "Fully Paid")
    # sprinkle some unresolved loans that must be filtered out
    unresolved = rng.random(rows) < .10
    loan_status = np.where(unresolved, rng.choice(["Current", "Late (31-120 days)", "In Grace Period"], rows), loan_status)

    df = pd.DataFrame({
        "id": np.arange(rows), "loan_amnt": loan_amnt, "funded_amnt": loan_amnt,
        "term": term, "int_rate": int_rate.round(2),
        "installment": (loan_amnt * (int_rate / 1200) / (1 - (1 + int_rate / 1200) ** -36)).round(2),
        "grade": grade, "sub_grade": sub_grade, "emp_title": "n/a", "emp_length": emp_length,
        "home_ownership": home, "annual_inc": annual_inc.round(0), "verification_status": verif,
        "issue_d": issue_d, "loan_status": loan_status, "purpose": purpose, "title": "n/a",
        "zip_code": "123xx", "addr_state": state, "dti": dti.round(2), "delinq_2yrs": delinq,
        "earliest_cr_line": earliest, "fico_range_low": fico_low, "fico_range_high": fico_low + 4,
        "inq_last_6mths": inq, "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal.round(0),
        "revol_util": revol_util.round(1), "total_acc": total_acc, "initial_list_status": init_list,
        # leakage columns, present so the pipeline proves it drops them
        "total_pymnt": np.where(bad, loan_amnt * .4, loan_amnt * 1.15), "recoveries": np.where(bad, 300, 0),
        "last_pymnt_amnt": rng.random(rows) * 500, "last_fico_range_low": fico_low - np.where(bad, 80, 0),
        "application_type": app_type, "mort_acc": mort_acc, "pub_rec_bankruptcies": bankr,
    })
    out = config.DATA_DIR / "sample_lending_club.csv"
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} synthetic rows -> {out}")
    print(f"  resolved default rate: {bad[~unresolved].mean():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60000)
    main(ap.parse_args().rows)
