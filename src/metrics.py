"""Credit-risk evaluation metrics and plots: ROC-AUC, Gini, KS, calibration."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov: max separation between cumulative bad and good distributions.
    Returns (KS, threshold at which it occurs)."""
    fpr, tpr, thr = roc_curve(y_true, p)
    i = np.argmax(tpr - fpr)
    return float(tpr[i] - fpr[i]), float(thr[i])


def summarise(y_true, p, name: str) -> dict:
    auc = roc_auc_score(y_true, p)
    ks, ks_thr = ks_statistic(np.asarray(y_true), np.asarray(p))
    return {"model": name, "auc": round(auc, 4), "gini": round(2 * auc - 1, 4),
            "ks": round(ks, 4), "ks_threshold": round(ks_thr, 4),
            "base_rate": round(float(np.mean(y_true)), 4), "n": int(len(y_true))}


def plot_roc(curves: dict[str, tuple], path: Path) -> None:
    plt.figure(figsize=(5.5, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC - out-of-time test set"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def plot_ks(y, p, name: str, path: Path) -> None:
    fpr, tpr, thr = roc_curve(y, p)
    i = np.argmax(tpr - fpr)
    plt.figure(figsize=(5.5, 4))
    plt.plot(thr, tpr, label="cum. bad captured"); plt.plot(thr, fpr, label="cum. good captured")
    plt.vlines(thr[i], fpr[i], tpr[i], colors="r", linestyles="--", label=f"KS = {tpr[i]-fpr[i]:.3f}")
    plt.xlim(0, min(1, float(np.nanmax(thr[np.isfinite(thr)])) * 1.05))
    plt.xlabel("predicted PD threshold"); plt.ylabel("cumulative share")
    plt.title(f"KS curve - {name}"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def plot_calibration(curves: dict[str, tuple], path: Path, n_bins: int = 10) -> None:
    plt.figure(figsize=(5.5, 5))
    for name, (y, p) in curves.items():
        frac, mean_p = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        plt.plot(mean_p, frac, "o-", label=name)
    lim = max(float(np.max(np.concatenate([p for _, p in curves.values()]))), .05)
    plt.plot([0, lim], [0, lim], "k--", lw=.8, label="perfect")
    plt.xlabel("mean predicted PD"); plt.ylabel("observed default rate")
    plt.title("Calibration (decile bins)"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def decile_table(y, p) -> pd.DataFrame:
    """Observed default rate by predicted-PD decile - a classic model-monitoring view."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11))
    return (d.groupby("decile", observed=True)
              .agg(n=("y", "size"), mean_pd=("p", "mean"), default_rate=("y", "mean"))
              .assign(lift=lambda t: t["default_rate"] / d["y"].mean()))


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2))
