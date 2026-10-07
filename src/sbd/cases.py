"""Реальные события 2024 года: паводки в Оренбургской, Курганской и Тюменской областях.

Проверяем три вещи:
  1) что происходило с тратами в названных в официальных сообщениях МО (по категориям);
  2) сколько этих МО детектор отмечает тревогой в апреле–июне и насколько это больше случайного
     (плацебо: случайные наборы МО того же размера в случайных трёхмесячных окнах);
  3) «подсказку новостей»: в регионах с действующим режимом ЧС порог тревоги снижается.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def residual_table(res_share: dict, ids, groups: dict[str, np.ndarray], dates, months) -> pd.DataFrame:
    """Среднее отклонение факта от прогноза (%) по группам МО, категориям и месяцам."""
    rows = []
    for gname, gidx in groups.items():
        for c, R in res_share.items():
            for p in months:
                t = dates.get_loc(pd.Period(p, "M"))
                rows.append({"группа": gname, "категория": c, "месяц": p,
                             "отклонение_%": 100 * float(np.nanmean(R[t, gidx])), "МО": len(gidx)})
    return pd.DataFrame(rows)


def alarm_test(score: np.ndarray, thr: float, idx: np.ndarray, t_start: int, win: int,
               t_range: tuple[int, int], draws: int, rng) -> dict:
    """Сколько МО из idx получили хотя бы одну тревогу в окне [t_start, t_start+win) и плацебо."""
    alarm = score >= thr
    obs = int(alarm[t_start: t_start + win, idx].any(axis=0).sum())
    n = score.shape[1]
    lo, hi = t_range
    null = np.empty(draws)
    for d in range(draws):
        t0 = rng.integers(lo, hi - win + 2)
        j = rng.choice(n, size=len(idx), replace=False)
        null[d] = alarm[t0: t0 + win, j].any(axis=0).sum()
    return {"МО в группе": len(idx), "МО с тревогой": obs,
            "ожидаемо случайно": float(null.mean()),
            "p_плацебо": float((np.sum(null >= obs) + 1) / (draws + 1))}


def guided_alarms(score: np.ndarray, thr_base: float, thr_guided: float, emerg: np.ndarray) -> np.ndarray:
    """Тревоги с «подсказкой новостей»: в МО, где действует режим ЧС в регионе, порог ниже.
    emerg — матрица (месяцы x МО) 0/1, построенная только по событиям, известным к концу месяца."""
    thr = np.where(emerg > 0, thr_guided, thr_base)
    return score >= thr


def top_alarms(score: np.ndarray, signed: np.ndarray, z_by_cat: dict, ref: pd.DataFrame, dates,
               t_from: int, k: int = 25) -> pd.DataFrame:
    rows = []
    S = np.nan_to_num(score, nan=-np.inf).copy()
    S[:t_from] = -np.inf
    flat = np.argsort(-S.ravel())[:k]
    cats = list(z_by_cat)
    for f in flat:
        t, i = np.unravel_index(f, S.shape)
        zc = {c: z_by_cat[c][t, i] for c in cats}
        drv = max(zc, key=lambda c: abs(zc[c]) if c != cats[0] else -1)
        rows.append({"месяц": str(dates[t]), "МО": ref.iloc[i]["name"], "регион": ref.iloc[i]["region"],
                     "направление": "всплеск" if signed[t, i] > 0 else "провал",
                     "оценка": round(float(score[t, i]), 1),
                     "z по всем категориям": round(float(zc[cats[0]]), 1),
                     "главная категория": drv, "z главной категории": round(float(zc[drv]), 1)})
    return pd.DataFrame(rows)
