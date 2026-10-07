"""Фундаментальная модель временных рядов Chronos-Bolt (Amazon, Apache-2.0), zero-shot на CPU.

Веса не входят в репозиторий. Положите папку модели (config.json + model.safetensors) в
models/chronos-bolt-small — `make foundation` скачает её с Hugging Face, если сеть доступна.
Проверка подлинности весов: SHA-256 model.safetensors должен совпадать с опубликованным на
huggingface.co/amazon/chronos-bolt-small (06a6a19b…a21dd).

Два варианта входа:
  chronos_bolt      — ряд отклонений МО от общего фактора (L - F); общую сезонность мы убираем
                      заранее и возвращаем через сезонный шаг фактора (как в factor_panel);
  chronos_bolt_raw  — сырой лог-ряд МО, модель сама ищет сезонность в 12–23 точках.
Прогноз — медиана (квантиль 0,5).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast import Factors

SHA256_SMALL = "06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd"


def available(model_dir: str) -> bool:
    p = Path(model_dir)
    return (p / "config.json").exists() and (p / "model.safetensors").exists()


def weights_sha256(model_dir: str) -> str:
    h = hashlib.sha256()
    with open(Path(model_dir) / "model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(model_dir: str):
    import torch
    from chronos import BaseChronosPipeline
    try:
        return BaseChronosPipeline.from_pretrained(model_dir, device_map="cpu", dtype=torch.float32)
    except TypeError:                                                 # старые версии transformers
        return BaseChronosPipeline.from_pretrained(model_dir, device_map="cpu", torch_dtype=torch.float32)


def _median(pipe, X: np.ndarray, H: int, batch: int) -> np.ndarray:
    """X: N x T (контекст), возвращает N x H медиан прогноза."""
    import torch
    out = []
    for b in range(0, X.shape[0], batch):
        ctx = torch.tensor(X[b: b + batch], dtype=torch.float32)
        q, _ = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5])
        out.append(q[:, :, 0].numpy())
    return np.vstack(out)


def backtest_chronos(panel: dict, ref: pd.DataFrame, nat: pd.DataFrame, cfg: dict, model_dir: str,
                     batch: int = 512) -> pd.DataFrame:
    import torch
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    pipe = _load(model_dir)
    fc = cfg["forecast"]
    cats = cfg["data"]["categories"]
    dates = panel[cats[0]].index
    t_from = dates.get_loc(pd.Period(fc["origins_from"], "M"))
    t_to = dates.get_loc(pd.Period(fc["origins_to"], "M"))
    last = len(dates) - 1
    rows = []
    for c in cats:
        fx = Factors(panel[c], ref.region, nat[cfg["data"]["national_map"][c]], fc["region_shrink_k"])
        D = fx.dev()
        ids = fx.ids
        for T in range(t_from, t_to + 1):
            H = last - T
            dev_hat = _median(pipe, D[: T + 1].T, H, batch)                 # отклонения от фактора
            raw_hat = _median(pipe, fx.L[: T + 1].T, H, batch)              # сырой лог-ряд
            for h in range(1, H + 1):
                step_f = fx.steps(T, h)[0]
                lp_rel = fx.F[T] + step_f + dev_hat[:, h - 1]
                y = np.exp(fx.L[T + h])
                for name, lp in (("chronos_bolt", lp_rel), ("chronos_bolt_raw", raw_hat[:, h - 1])):
                    rows.append(pd.DataFrame({"category": c, "territory_id": ids, "origin": str(dates[T]),
                                              "h": h, "target": str(dates[T + h]), "model": name,
                                              "yhat": np.exp(lp), "y": y}))
        print(f"  chronos {c}", flush=True)
    return pd.concat(rows, ignore_index=True)
