"""
Step 4 - Convert the logistic model into an industry-standard points scorecard.

Standard scaling (Siddiqi, "Credit Risk Scorecards"):
    score = offset + factor * ln(odds_good)
    factor = PDO / ln(2)
    offset = BASE_SCORE - factor * ln(BASE_ODDS)

With a logistic model  ln(odds_bad) = a + sum_i b_i * WoE_i,  so ln(odds_good) = -(a + sum b_i WoE_i) and
    points for feature i in bin j = -(b_i * WoE_ij + a/n) * factor + offset/n

Outputs:
  outputs/scorecard.csv            - points per feature/bin (the actual deliverable)
  outputs/score_bands.csv          - observed default rate by score band on test
  outputs/cutoff_analysis.csv      - approval rate vs expected default rate for each cutoff
  outputs/score_distribution.png, outputs/cutoff_tradeoff.png

Usage:  python src/scorecard.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main() -> None:
    with open(config.OUT_DIR / "logistic_model.pkl", "rb") as f:
        art = pickle.load(f)
    lr, binner, feats = art["model"], art["binner"], art["features"]
    num, cat = art["numeric"], art["categorical"]

    factor = config.PDO / np.log(2)
    offset = config.BASE_SCORE - factor * np.log(config.BASE_ODDS)
    a, b = lr.intercept_[0], dict(zip(feats, lr.coef_[0]))
    n = len(feats)
    print(f"factor = {factor:.3f}   offset = {offset:.3f}   (PDO={config.PDO}, {config.BASE_SCORE} @ {config.BASE_ODDS}:1)")

    # ---- points table ------------------------------------------------------
    rows = []
    for col in feats:
        tbl = binner.woe_tables[col]
        for bin_label, r in tbl.iterrows():
            pts = -(b[col] * r["woe"] + a / n) * factor + offset / n
            rows.append({"feature": col, "bin": str(bin_label), "n_train": int(r["n"]),
                         "bad_rate_train": round(r["bad_rate"], 4), "woe": round(r["woe"], 4),
                         "points": int(round(pts))})
    card = pd.DataFrame(rows)
    card.to_csv(config.OUT_DIR / "scorecard.csv", index=False)
    print(f"\nscorecard: {len(card)} rows across {n} features -> outputs/scorecard.csv")
    print(card.groupby("feature")["points"].agg(["min", "max"]).assign(range=lambda d: d["max"] - d["min"])
              .sort_values("range", ascending=False).to_string())

    # ---- score each test loan ---------------------------------------------
    df = pd.read_pickle(config.PREPARED)
    te = df[df["is_test"] == 1]
    labels = binner.bin_labels(te, num, cat)
    points_map = {(r.feature, r.bin): r.points for r in card.itertuples()}
    neutral = {c: int(round(-(a / n) * factor + offset / n)) for c in feats}  # unseen bin
    score = np.zeros(len(te), dtype=int)
    for col in feats:
        score += labels[col].map(lambda v, c=col: points_map.get((c, str(v)), neutral[c])).astype(int).values
    te = te.assign(score=score)

    # sanity: score should be monotone with the model's PD
    X_te = binner.transform(te, num, cat)
    te["pd"] = lr.predict_proba(X_te)[:, 1]
    print(f"\ncorr(score, ln-odds) = {np.corrcoef(te['score'], -np.log(te['pd']/(1-te['pd'])))[0,1]:.4f}  (should be ~1.0)")

    # ---- score bands -------------------------------------------------------
    edges = list(range(int(te["score"].min() // 20 * 20), int(te["score"].max() // 20 * 20 + 40), 20))
    te["band"] = pd.cut(te["score"], bins=edges, right=False)
    bands = (te.groupby("band", observed=True)
               .agg(n=("score", "size"), default_rate=(config.TARGET, "mean"))
               .assign(pct_of_book=lambda d: d["n"] / d["n"].sum()))
    bands.to_csv(config.OUT_DIR / "score_bands.csv")
    print("\nobserved default rate by score band (test):\n", bands.round(4).to_string())

    # ---- cutoff analysis ---------------------------------------------------
    cuts = np.arange(te["score"].quantile(.02), te["score"].quantile(.98), 5).astype(int)
    rows = []
    for c in cuts:
        approved = te[te["score"] >= c]
        rows.append({"cutoff": int(c), "approval_rate": len(approved) / len(te),
                     "expected_default_rate": approved[config.TARGET].mean() if len(approved) else np.nan,
                     "bad_rate_declined": te.loc[te["score"] < c, config.TARGET].mean() if (te["score"] < c).any() else np.nan})
    cut_tbl = pd.DataFrame(rows)
    cut_tbl.to_csv(config.OUT_DIR / "cutoff_analysis.csv", index=False)
    base = te[config.TARGET].mean()
    # example decision: the cutoff that halves the portfolio default rate
    target_rate = base / 2
    pick = cut_tbl[cut_tbl["expected_default_rate"] <= target_rate].iloc[0] if (cut_tbl["expected_default_rate"] <= target_rate).any() else cut_tbl.iloc[-1]
    print(f"\nportfolio default rate (approve-all): {base:.2%}")
    print(f"cutoff {int(pick.cutoff)} -> approve {pick.approval_rate:.1%} of applicants at {pick.expected_default_rate:.2%} expected default "
          f"(declined pool default rate {pick.bad_rate_declined:.2%})")

    # ---- plots -------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    te.groupby(config.TARGET)["score"].plot(kind="hist", bins=40, alpha=.55, density=True,
                                            legend=True, label=None)
    plt.legend(["good", "bad"]); plt.xlabel("score"); plt.title("Score distribution - test")
    plt.tight_layout(); plt.savefig(config.OUT_DIR / "score_distribution.png", dpi=150); plt.close()

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(cut_tbl["cutoff"], cut_tbl["approval_rate"], color="tab:blue", label="approval rate")
    ax1.set_xlabel("score cutoff"); ax1.set_ylabel("approval rate", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(cut_tbl["cutoff"], cut_tbl["expected_default_rate"], color="tab:red", label="expected default rate")
    ax2.set_ylabel("expected default rate", color="tab:red")
    ax1.axvline(pick.cutoff, color="k", ls="--", lw=.8)
    plt.title("Approval vs default trade-off"); fig.tight_layout()
    plt.savefig(config.OUT_DIR / "cutoff_tradeoff.png", dpi=150); plt.close()
    print(f"\nsaved scorecard outputs and plots to {config.OUT_DIR}")


if __name__ == "__main__":
    main()
