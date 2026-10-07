"""Событийный (новостной) слой: решения Банка России и режимы ЧС.

Правило согласования с данными СберИндекса: событие относится к месяцу m, только если
дата, когда о нём стало известно (date_known), не позже последнего дня m. Для прогноза из
точки T используются события с date_known <= конец месяца T.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

RATE_BEFORE_2023 = 7.5          # ключевая ставка, действовавшая на 01.01.2023 (с 19.09.2022)
EMERGENCY_ACTIVE_MONTHS = 3     # сколько месяцев (включая месяц объявления) считаем режим ЧС действующим


def load_events(path: str) -> pd.DataFrame:
    ev = pd.read_csv(path)
    ev["date_known"] = pd.to_datetime(ev["date_known"])
    ev["month"] = ev["date_known"].dt.to_period("M")
    return ev


def key_rate_by_month(ev: pd.DataFrame, months: pd.PeriodIndex) -> pd.Series:
    """Ключевая ставка на конец каждого месяца (по решениям, известным к концу месяца)."""
    r = ev[ev.event_type == "cbr_rate"].sort_values("date_known")
    r = r.assign(rate=r.detail.map(lambda s: float(re.findall(r"(\d+\.\d+)", s)[-1])))
    out = {}
    for p in months:
        known = r[r.date_known <= p.to_timestamp(how="end")]
        out[p] = known.rate.iloc[-1] if len(known) else RATE_BEFORE_2023
    return pd.Series(out)


def match_municipalities(ev: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Сопоставляет названия МО из событий с territory_id (внутри региона, по короткому имени)."""
    rows = []
    for _, e in ev[ev.municipality.notna()].iterrows():
        cand = ref[(ref.region == e.region) & (ref.short == e.municipality)]
        rows.append({"event_type": e.event_type, "region": e.region, "municipality": e.municipality,
                     "month": e.month, "verified": e.verified,
                     "territory_id": int(cand.index[0]) if len(cand) == 1 else np.nan,
                     "in_panel": len(cand) == 1})
    return pd.DataFrame(rows)


def event_features(ev: pd.DataFrame, ref: pd.DataFrame, months: pd.PeriodIndex) -> dict:
    """Признаки для прогноза: изменение ставки за 3 мес. и действующий режим ЧС в регионе МО."""
    all_months = pd.period_range(months[0] - 3, months[-1], freq="M")
    rate = key_rate_by_month(ev, all_months)
    rate_chg = {p: float(rate[p] - rate[p - 3]) for p in months}
    reg_ev = ev[ev.event_type.isin(["emergency_regional", "emergency_federal"])]
    regions = ref.region.values
    emerg = {}
    for p in months:
        active = reg_ev[(reg_ev.month <= p) & (p <= reg_ev.month + (EMERGENCY_ACTIVE_MONTHS - 1))]
        emerg[p] = np.isin(regions, active.region.unique()).astype(float)
    return {"rate_chg_3m": rate_chg, "emerg_region": emerg, "rate": rate}
