"""Метрики прогноза и статистические тесты."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def point_metrics(y: np.ndarray, yhat: np.ndarray, y0: np.ndarray | None = None) -> dict:
    """MAE (руб.), WAPE, R² уровня и R² изменения относительно уровня в точке прогноза y0."""
    e = y - yhat
    out = {"MAE": float(np.mean(np.abs(e))),
           "WAPE": float(np.sum(np.abs(e)) / np.sum(np.abs(y))),
           "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}
    if y0 is not None:
        d = y - y0
        out["R2_change"] = float(1 - np.sum(e ** 2) / np.sum((d - d.mean()) ** 2))
    return out


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]:
    """Диболд–Мариано с поправкой Харви–Лейборна–Ньюболда. loss_* — ряды потерь по точкам прогноза.
    Отрицательная статистика: у модели a потери меньше."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    if n < 3:
        return np.nan, np.nan
    dc = d - d.mean()
    lags = min(h - 1, n - 2)
    gamma = [np.sum(dc[k:] * dc[: n - k]) / n for k in range(lags + 1)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        var = gamma[0] / n
    dm = d.mean() / np.sqrt(var)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) if n + 1 - 2 * h > 0 else 1.0
    stat = dm * corr
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def wilcoxon_mo(err_a: np.ndarray, err_b: np.ndarray) -> float:
    """Парный знаково-ранговый тест Уилкоксона по МО (средние абсолютные ошибки МО)."""
    return float(stats.wilcoxon(err_a, err_b, alternative="less").pvalue)


def summarize(bt: pd.DataFrame, baseline: str, horizons=None) -> pd.DataFrame:
    """Сводка по (категория, горизонт, модель) + сравнение с базовой моделью."""
    bt = bt.copy()
    bt["ae"] = (bt.y - bt.yhat).abs()
    y0 = (bt[bt.model == "naive"].set_index(["category", "territory_id", "origin", "h"]).yhat)
    rows = []
    groups = bt.groupby(["category", "h", "model"]) if horizons is None else \
        bt[bt.h.isin(horizons)].groupby(["category", "h", "model"])
    for (c, h, m), g in groups:
        key = pd.MultiIndex.from_frame(g[["category", "territory_id", "origin", "h"]])
        r = point_metrics(g.y.values, g.yhat.values, y0.reindex(key).values)
        r.update({"category": c, "h": h, "model": m, "n_forecasts": len(g),
                  "n_origins": g.origin.nunique()})
        if m != baseline:
            b = bt[(bt.category == c) & (bt.h == h) & (bt.model == baseline)]
            j = g.merge(b[["territory_id", "origin", "ae"]], on=["territory_id", "origin"],
                        suffixes=("", "_b"))
            la = j.groupby("origin").ae.mean().values
            lb = j.groupby("origin").ae_b.mean().values
            r["MAE_vs_base_%"] = float(100 * (j.ae.mean() / j.ae_b.mean() - 1))
            r["DM_stat"], r["DM_p"] = dm_test(la, lb, h)
            mo = j.groupby("territory_id")[["ae", "ae_b"]].mean()
            r["share_MO_better"] = float((mo.ae < mo.ae_b).mean())
            r["wilcoxon_p"] = wilcoxon_mo(mo.ae.values, mo.ae_b.values)
        rows.append(r)
    return pd.DataFrame(rows)
