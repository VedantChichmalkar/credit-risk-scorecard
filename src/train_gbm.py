"""
Step 3 - Challenger model: gradient boosting on raw (un-binned) features.

Uses LightGBM if installed, otherwise scikit-learn's HistGradientBoostingClassifier
(same algorithm family; results are comparable). Both handle NaNs natively.
Compared against the logistic champion on the same out-of-time test set.

Usage:  python src/train_gbm.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import metrics  # noqa: E402

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_LGB = False


def main() -> None:
    df = pd.read_pickle(config.PREPARED)
    num = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in config.CATEGORICAL_FEATURES if c in df.columns]
    for c in cat:
        df[c] = df[c].astype("category")
    tr, te = df[df["is_test"] == 0], df[df["is_test"] == 1]
    X_tr, y_tr = tr[num + cat], tr[config.TARGET]
    X_te, y_te = te[num + cat], te[config.TARGET]

    if HAVE_LGB:
        print("using LightGBM")
        model = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.02, num_leaves=31, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
            random_state=config.RANDOM_STATE, verbose=-1)
        # early stopping on a time-ordered tail of the TRAIN set (never on test)
        cut = int(len(tr) * 0.85)
        order = np.argsort(tr["issue_year"].values, kind="stable")
        fit_idx, val_idx = order[:cut], order[cut:]
        model.fit(X_tr.iloc[fit_idx], y_tr.iloc[fit_idx],
                  eval_set=[(X_tr.iloc[val_idx], y_tr.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        importance = pd.Series(model.feature_importances_, index=num + cat)
    else:
        print("LightGBM not installed - using sklearn HistGradientBoostingClassifier")
        model = HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
            l2_regularization=5.0, early_stopping=True, validation_fraction=0.15,
            categorical_features=[X_tr.columns.get_loc(c) for c in cat],
            random_state=config.RANDOM_STATE)
        model.fit(X_tr, y_tr)
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(model, X_te.sample(min(20000, len(X_te)), random_state=0),
                                    y_te.loc[X_te.sample(min(20000, len(X_te)), random_state=0).index],
                                    n_repeats=3, random_state=0, scoring="roc_auc")
        importance = pd.Series(pi.importances_mean, index=num + cat)

    p_tr, p_te = model.predict_proba(X_tr)[:, 1], model.predict_proba(X_te)[:, 1]
    importance.sort_values(ascending=False).to_csv(config.OUT_DIR / "gbm_feature_importance.csv")
    print("\ntop features:\n", importance.sort_values(ascending=False).head(12).round(4).to_string())

    res_tr = metrics.summarise(y_tr, p_tr, "gbm (train)")
    res_te = metrics.summarise(y_te, p_te, "gbm (test, out-of-time)")
    print("\n", pd.DataFrame([res_tr, res_te]).to_string(index=False))
    metrics.save_json({"train": res_tr, "test": res_te, "backend": "lightgbm" if HAVE_LGB else "sklearn_hgb"},
                      config.OUT_DIR / "gbm_metrics.json")
    metrics.decile_table(y_te, p_te).to_csv(config.OUT_DIR / "gbm_decile_table.csv")
    metrics.plot_ks(y_te, p_te, "gbm", config.OUT_DIR / "ks_gbm.png")

    # ---- champion vs challenger plots ----------------------------------
    lg = pd.read_pickle(config.OUT_DIR / "test_predictions_logistic.pkl")
    curves = {"logistic (WoE)": (lg["y"].values, lg["p_logistic"].values), "gbm": (y_te.values, p_te)}
    metrics.plot_roc(curves, config.OUT_DIR / "roc_comparison.png")
    metrics.plot_calibration(curves, config.OUT_DIR / "calibration_comparison.png")

    with open(config.OUT_DIR / "gbm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\nsaved GBM model, metrics and comparison plots to {config.OUT_DIR}")


if __name__ == "__main__":
    main()
