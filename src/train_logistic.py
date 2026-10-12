"""
Step 2 - Champion scorecard model: WoE-binned logistic regression.

  * bins learned on train only
  * features with IV < IV_THRESHOLD dropped
  * logistic regression on WoE-transformed features
  * evaluated on the out-of-time test set (AUC, Gini, KS, calibration)

Usage:  python src/train_logistic.py
"""
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import metrics  # noqa: E402
from src.woe import WoeBinner  # noqa: E402


def main() -> None:
    df = pd.read_pickle(config.PREPARED)
    num = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]
    tr, te = df[df["is_test"] == 0], df[df["is_test"] == 1]
    y_tr, y_te = tr[config.TARGET], te[config.TARGET]

    # ---- WoE binning on train -------------------------------------------
    binner = WoeBinner(config.N_BINS, config.MIN_BIN_FRAC).fit(tr, y_tr, num, cat)
    iv = binner.iv_table()
    iv.to_csv(config.OUT_DIR / "iv_table.csv")
    print("Information Value by feature:\n", iv.round(4).to_string())

    selected = [c for c in num + cat if binner.iv[c] >= config.IV_THRESHOLD]
    suspicious = [c for c in selected if binner.iv[c] > 0.5]
    if suspicious:
        print(f"\n!! IV > 0.5 for {suspicious} - check these for leakage before trusting results")
    print(f"\nselected {len(selected)}/{len(num)+len(cat)} features with IV >= {config.IV_THRESHOLD}")
    # ---- drop near-duplicate features (keep the higher-IV one of each pair) ----
    X_all = binner.transform(tr, [c for c in selected if c in num], [c for c in selected if c in cat])
    corr = X_all.corr().abs()
    ordered = sorted(selected, key=lambda c: -binner.iv[c])
    kept: list[str] = []
    for c in ordered:
        if all(corr.loc[c, k] < config.CORR_THRESHOLD for k in kept):
            kept.append(c)
        else:
            print(f"  dropping {c}: |corr| >= {config.CORR_THRESHOLD} with an already-kept, higher-IV feature")
    selected = kept
    sel_num, sel_cat = [c for c in selected if c in num], [c for c in selected if c in cat]
    print(f"final feature set ({len(selected)}): {selected}")

    X_tr = binner.transform(tr, sel_num, sel_cat)
    X_te = binner.transform(te, sel_num, sel_cat)

    # ---- logistic regression --------------------------------------------
    # C is mild L2 regularisation; WoE features are already on a comparable scale.
    lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    lr.fit(X_tr, y_tr)
    p_tr, p_te = lr.predict_proba(X_tr)[:, 1], lr.predict_proba(X_te)[:, 1]

    coefs = pd.Series(lr.coef_[0], index=selected, name="coef").sort_values()
    print("\nlogistic coefficients (on WoE):\n", coefs.round(4).to_string())

    # ---- evaluation -----------------------------------------------------
    res_tr = metrics.summarise(y_tr, p_tr, "logistic_woe (train)")
    res_te = metrics.summarise(y_te, p_te, "logistic_woe (test, out-of-time)")
    print("\n", pd.DataFrame([res_tr, res_te]).to_string(index=False))
    metrics.save_json({"train": res_tr, "test": res_te}, config.OUT_DIR / "logistic_metrics.json")
    metrics.decile_table(y_te, p_te).to_csv(config.OUT_DIR / "logistic_decile_table.csv")
    metrics.plot_ks(y_te, p_te, "logistic_woe", config.OUT_DIR / "ks_logistic.png")

    # ---- persist for scorecard + comparison ----------------------------
    with open(config.OUT_DIR / "logistic_model.pkl", "wb") as f:
        pickle.dump({"model": lr, "binner": binner, "features": selected,
                     "numeric": sel_num, "categorical": sel_cat}, f)
    pd.DataFrame({"y": y_te.values, "p_logistic": p_te}, index=te.index).to_pickle(
        config.OUT_DIR / "test_predictions_logistic.pkl")
    print(f"\nsaved model + predictions to {config.OUT_DIR}")


if __name__ == "__main__":
    main()
