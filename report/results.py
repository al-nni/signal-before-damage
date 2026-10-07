"""Собирает все числа для отчёта и презентации из outputs/ в один словарь R.

Любое число в тексте берётся отсюда — поэтому отчёт всегда совпадает с расчётом.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import os

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("SBD_OUT", ROOT / "outputs"))
TOTAL = "Все категории"
CATS = ["Все категории", "Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт"]


def load() -> dict:
    R: dict = {}
    R["data"] = json.load(open(OUT / "data_report.json"))
    summ = pd.read_csv(OUT / "table_forecast.csv")
    R["summ"] = summ
    R["best"] = json.load(open(OUT / "best_model.json"))
    t = summ[summ.category == TOTAL].set_index(["model", "h"])
    R["tot"] = t
    R["models_present"] = sorted(summ.model.unique())
    # средняя по категориям MAE и выигрыш к Prophet
    main = json.load(open(OUT / "best_model.json"))["best"]
    main = main if main in ("ensemble", "ensemble_fm") else "ensemble"
    R["avg_gain"] = (summ[summ.model == main].groupby("h")["MAE_vs_base_%"].mean()).to_dict()
    R["vs_log_y"] = pd.read_csv(OUT / "table_forecast_vs_prophet_log_y.csv").set_index(["model", "h"])
    det = pd.read_csv(OUT / "table_detect.csv", index_col=0)
    R["det"] = det
    R["det_rep"] = pd.read_csv(OUT / "detect_by_repeat.csv")
    R["main_det"] = json.load(open(OUT / "cases" / "main_detector.json"))["main_detector"]
    R["flood_res"] = pd.read_csv(OUT / "cases" / "flood_residuals.csv")
    R["flood_mo"] = pd.read_csv(OUT / "cases" / "flood_mo_alarms.csv")
    R["flood_test"] = pd.read_csv(OUT / "cases" / "flood_alarm_test.csv")
    R["guided"] = pd.read_csv(OUT / "cases" / "news_guided.csv")
    R["top"] = pd.read_csv(OUT / "cases" / "top_alarms.csv")
    R["events"] = pd.read_csv(ROOT / "data" / "events" / "events.csv")
    R["chronos"] = "chronos_bolt" in R["models_present"]
    # главная модель отчёта — лучший из заранее заданных ансамблей по средней MAE (best_model.json)
    R["main"] = R["best"]["best"] if R["best"]["best"] in ("ensemble", "ensemble_fm") else "ensemble"
    R["foundation"] = json.load(open(OUT / "foundation_model.json")) if (OUT / "foundation_model.json").exists() else {}
    return R


def g(R, model, h, col="MAE"):
    """Метрика по «Все категории» для модели и горизонта."""
    return float(R["tot"].loc[(model, h), col])


def all_better_h(R, model):
    """Горизонты, на которых модель лучше Prophet во всех шести категориях."""
    return [h for h in (1, 3, 6, 12) if all(v > 0 for v in gain_cat(R, model, h).values())]


def gain_cat(R, model, h):
    s = R["summ"]
    return {c: -float(s[(s.category == c) & (s.model == model) & (s.h == h)]["MAE_vs_base_%"].iloc[0]) for c in CATS}


def det(R, name, col):
    return float(R["det"].loc[name, f"{col}__mean"])


def det_sd(R, name, col):
    return float(R["det"].loc[name, f"{col}__std"])
