"""Prophet — базовая модель для сравнения (по каждому МО отдельно, без утечки).

Два варианта:
  prophet        — настройки по умолчанию на уровне ряда (как «из коробки»);
  prophet_log_y  — на логарифме с принудительной годовой сезонностью (Фурье 3-го порядка, L-BFGS).
Прогнозы кэшируются в outputs/cache, поэтому повторный запуск быстрый.
"""
from __future__ import annotations

import logging
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ("prophet", "prophet_log_y")


def _quiet():
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").disabled = True


def _fit_one(args):
    variant, ds_train, y_train, n_ahead = args
    _quiet()
    from prophet import Prophet
    if variant == "prophet":
        m = Prophet(uncertainty_samples=0)          # интервалы не нужны: ускоряет predict, yhat тот же
        y = y_train
    else:
        m = Prophet(yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False,
                    uncertainty_samples=0)
        y = np.log(y_train)
    # для варианта с годовой сезонностью оптимизатор Newton (выбор Prophet для коротких рядов)
    # в ~15 раз медленнее, поэтому задаём L-BFGS; вариант «по умолчанию» не трогаем
    kw = {} if variant == "prophet" else {"algorithm": "LBFGS"}
    m.fit(pd.DataFrame({"ds": ds_train, "y": y}), **kw)
    fut = pd.DataFrame({"ds": pd.date_range(ds_train[-1] + pd.offsets.MonthBegin(1),
                                            periods=n_ahead, freq="MS")})
    yhat = m.predict(fut)["yhat"].values
    return np.exp(yhat) if variant != "prophet" else yhat


def _run_chunk(args):
    variant, W_values, cols, dates, origins = args
    _quiet()
    out = []
    last = len(dates) - 1
    for j, col in enumerate(cols):
        for T in origins:
            n_ahead = last - T
            ds = dates[: T + 1].to_timestamp()
            yhat = _fit_one((variant, ds, W_values[: T + 1, j], n_ahead))
            for k in range(n_ahead):
                out.append((col, T, k + 1, yhat[k]))
    return out


def run(panel: dict, cfg: dict, variant: str, categories) -> pd.DataFrame:
    cache = Path(cfg["paths"]["out"]) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for c in categories:
        f = cache / f"{variant}__{c}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
            continue
        W = panel[c]
        dates = W.index
        o_from = dates.get_loc(pd.Period(cfg["forecast"]["origins_from"], "M"))
        o_to = dates.get_loc(pd.Period(cfg["forecast"]["origins_to"], "M"))
        origins = list(range(o_from, o_to + 1))
        cols = list(W.columns)
        n_jobs = cfg["forecast"]["prophet"]["n_jobs"]
        chunks = np.array_split(np.arange(len(cols)), n_jobs * 8)
        tasks = [(variant, W.values[:, ch], [cols[i] for i in ch], dates, origins) for ch in chunks]
        with Pool(n_jobs) as p:
            res = p.map(_run_chunk, tasks)
        rows = [r for part in res for r in part]
        df = pd.DataFrame(rows, columns=["territory_id", "origin_idx", "h", "yhat"])
        df["origin"] = [str(dates[i]) for i in df.origin_idx]
        df["target"] = [str(dates[i + h]) for i, h in zip(df.origin_idx, df.h)]
        df["category"] = c
        df["model"] = variant
        df = df.drop(columns="origin_idx")
        df.to_parquet(f)
        frames.append(df)
        print(f"  prophet[{variant}] {c}: {len(df)} прогнозов", flush=True)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from sbd.data import build_panel, load_config
    cfg = load_config()
    panel, _ = build_panel(cfg)
    variant = sys.argv[1] if len(sys.argv) > 1 else "prophet"
    cats = sys.argv[2].split("|") if len(sys.argv) > 2 else cfg["data"]["categories"]
    run(panel, cfg, variant, cats)
