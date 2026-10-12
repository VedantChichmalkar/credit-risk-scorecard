"""
Weight-of-Evidence (WoE) binning and Information Value (IV).

    WoE_bin = ln( %good_in_bin / %bad_in_bin )
    IV      = sum_bins (%good - %bad) * WoE_bin

Rule of thumb for IV:  <0.02 useless | 0.02-0.1 weak | 0.1-0.3 medium | 0.3-0.5 strong | >0.5 suspicious (check leakage)

Bins are learned on the TRAIN set only and then applied to test, so there is no
information leak from the test vintages.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 0.5  # Laplace-style smoothing so empty bins don't blow up the log


@dataclass
class WoeBinner:
    n_bins: int = 10
    min_bin_frac: float = 0.02
    # learned state
    numeric_edges: dict[str, np.ndarray] = field(default_factory=dict)
    categorical_levels: dict[str, set] = field(default_factory=dict)
    woe_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    iv: dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- binning
    def _bin_numeric(self, s: pd.Series, col: str, fit: bool) -> pd.Series:
        if fit:
            qs = np.linspace(0, 1, self.n_bins + 1)
            edges = np.unique(s.dropna().quantile(qs).values)
            edges[0], edges[-1] = -np.inf, np.inf
            self.numeric_edges[col] = edges
        edges = self.numeric_edges[col]
        binned = pd.cut(s, bins=edges, include_lowest=True).astype("string")
        return binned.fillna("MISSING")

    def _bin_categorical(self, s: pd.Series, col: str, fit: bool) -> pd.Series:
        s = s.astype("string").fillna("MISSING")
        if fit:
            freq = s.value_counts(normalize=True)
            self.categorical_levels[col] = set(freq[freq >= self.min_bin_frac].index)
        keep = self.categorical_levels[col]
        return s.where(s.isin(keep), "OTHER")

    # ------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str]) -> "WoeBinner":
        total_good, total_bad = (y == 0).sum(), (y == 1).sum()
        for col in numeric + categorical:
            binned = (self._bin_numeric(X[col], col, fit=True) if col in numeric
                      else self._bin_categorical(X[col], col, fit=True))
            tbl = (pd.DataFrame({"bin": binned, "y": y.values})
                     .groupby("bin", observed=True)["y"]
                     .agg(n="size", bad="sum"))
            tbl["good"] = tbl["n"] - tbl["bad"]
            tbl["pct_good"] = (tbl["good"] + EPS) / (total_good + EPS * len(tbl))
            tbl["pct_bad"] = (tbl["bad"] + EPS) / (total_bad + EPS * len(tbl))
            tbl["bad_rate"] = tbl["bad"] / tbl["n"]
            tbl["woe"] = np.log(tbl["pct_good"] / tbl["pct_bad"])
            tbl["iv_contrib"] = (tbl["pct_good"] - tbl["pct_bad"]) * tbl["woe"]
            self.woe_tables[col] = tbl
            self.iv[col] = float(tbl["iv_contrib"].sum())
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in numeric + categorical:
            binned = (self._bin_numeric(X[col], col, fit=False) if col in numeric
                      else self._bin_categorical(X[col], col, fit=False))
            mapping = self.woe_tables[col]["woe"]
            out[col] = binned.map(mapping).astype(float).fillna(0.0)  # unseen bin -> neutral
        return out

    def bin_labels(self, X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
        """Return the bin label (not WoE) for each row - used to build the scorecard."""
        out = pd.DataFrame(index=X.index)
        for col in numeric + categorical:
            out[col] = (self._bin_numeric(X[col], col, fit=False) if col in numeric
                        else self._bin_categorical(X[col], col, fit=False))
        return out

    def iv_table(self) -> pd.DataFrame:
        return (pd.Series(self.iv, name="IV").sort_values(ascending=False)
                  .to_frame().assign(strength=lambda d: pd.cut(
                      d["IV"], [-np.inf, .02, .1, .3, .5, np.inf],
                      labels=["useless", "weak", "medium", "strong", "suspicious"])))
