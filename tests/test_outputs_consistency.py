"""Сверка: числа в outputs/ пересчитываются независимым кодом из сырых прогнозов,
а ключевые числа присутствуют в PDF отчёта. Пропускается, если расчёт ещё не запускали."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
need = pytest.mark.skipif(not (OUT / "backtest.parquet").exists(), reason="нет outputs/backtest.parquet")


def ru(x, d=0):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",").replace("-", "−")


@need
def test_mae_table_matches_raw_forecasts():
    bt = pd.read_parquet(OUT / "backtest.parquet", filters=[("category", "==", "Все категории")])
    summ = pd.read_csv(OUT / "table_forecast.csv")
    for model in ["prophet", "ensemble", "factor_region", "snaive_drift"]:
        for h in (1, 3, 6, 12):
            g = bt[(bt.model == model) & (bt.h == h)]
            mae = float(np.mean(np.abs(g.y.values - g.yhat.values)))
            ref = summ[(summ.category == "Все категории") & (summ.model == model) & (summ.h == h)].MAE.iloc[0]
            assert abs(mae - ref) < 1e-6, (model, h, mae, ref)
            n_orig = {1: 12, 3: 10, 6: 7, 12: 1}[h]
            assert g.origin.nunique() == n_orig and len(g) == 2016 * n_orig


@need
def test_ensemble_is_geometric_mean_of_members():
    bt = pd.read_parquet(OUT / "backtest.parquet", filters=[("category", "==", "Продовольствие"), ("h", "==", 3)])
    p = bt.pivot_table(index=["territory_id", "origin"], columns="model", values="yhat")
    expect = np.exp(np.log(p[["factor_region", "snaive_drift"]]).mean(axis=1))
    np.testing.assert_allclose(p["ensemble"].values, expect.values, rtol=1e-10)
    if "chronos_bolt" in p:
        expect = np.exp(np.log(p[["factor_region", "snaive_drift", "chronos_bolt"]]).mean(axis=1))
        np.testing.assert_allclose(p["ensemble_fm"].values, expect.values, rtol=1e-10)


@need
def test_detection_table_matches_repeats():
    rep = pd.read_csv(OUT / "detect_by_repeat.csv")
    tab = pd.read_csv(OUT / "table_detect.csv", index_col=0)
    m = rep.groupby("detector")["recall@3"].mean()
    for d, v in m.items():
        assert abs(tab.loc[d, "recall@3__mean"] - v) < 1e-12
    assert rep.repeat.nunique() == 5


@need
def test_key_numbers_are_in_report_pdf():
    import pdfplumber
    pdf = ROOT / "report" / "report_signal_before_damage.pdf"
    text = "\n".join(p.extract_text() or "" for p in pdfplumber.open(pdf).pages).replace(" ", " ")
    summ = pd.read_csv(OUT / "table_forecast.csv")
    t = summ[summ.category == "Все категории"].set_index(["model", "h"])
    for key in [("ensemble", 1), ("prophet", 1), ("ensemble", 3), ("prophet", 12)]:
        assert ru(t.loc[key, "MAE"]) in text, key
    main = json.load(open(OUT / "cases" / "main_detector.json"))["main_detector"]
    tab = pd.read_csv(OUT / "table_detect.csv", index_col=0)
    assert ru(100 * tab.loc[main, "recall@3__mean"]) + "%" in text
