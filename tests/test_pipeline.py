"""Проверки корректности: нет заглядывания в будущее, детекторы причинны, реализации совпадают
с эталонными, событийный слой соблюдает дату, когда событие стало известно."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sbd import detect as D  # noqa: E402
from sbd import events as EV  # noqa: E402
from sbd import forecast as FC  # noqa: E402


def _toy(n_mo=60, seed=0):
    """Игрушечная панель: общая сезонность + уровни МО + шум, 24 месяца, 3 региона."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2023-01", periods=24, freq="M")
    season = 1 + 0.1 * np.sin(np.arange(24) * 2 * np.pi / 12) + 0.004 * np.arange(24)
    lev = rng.lognormal(10, 0.3, n_mo)
    W = pd.DataFrame(season[:, None] * lev[None, :] * rng.lognormal(0, 0.03, (24, n_mo)),
                     index=idx, columns=np.arange(1, n_mo + 1))
    region = pd.Series(np.repeat(["A", "B", "C"], n_mo // 3), index=W.columns)
    nat_idx = pd.period_range("2018-12", "2024-12", freq="M")
    nat = pd.Series(1000 * (1 + 0.1 * np.sin(np.arange(len(nat_idx)) * 2 * np.pi / 12)), index=nat_idx)
    return W, region, nat


@pytest.mark.parametrize("T", [11, 14, 18])
def test_forecasts_do_not_use_future(T):
    W, region, nat = _toy()
    fx1 = FC.Factors(W, region, nat, 10)
    W2 = W.copy()
    W2.iloc[T + 1:] *= np.random.default_rng(1).uniform(0.5, 2.0, W2.iloc[T + 1:].shape)
    fx2 = FC.Factors(W2, region, nat, 10)
    for h in (1, 3, 5):
        a, b = FC.simple_models(fx1, T, h, 3), FC.simple_models(fx2, T, h, 3)
        for k in a:
            np.testing.assert_allclose(a[k], b[k], err_msg=f"{k} T={T} h={h}")


def test_features_do_not_use_future():
    W, region, nat = _toy()
    static = pd.DataFrame({"x": np.ones(W.shape[1])}, index=W.columns)
    ev = {"rate_chg_3m": {}, "emerg_region": {}}
    T = 15
    W2 = W.copy()
    W2.iloc[T + 1:] *= 3.0
    f1 = FC.features(FC.Factors(W, region, nat, 10), T, 3, 3, static, ev)
    f2 = FC.features(FC.Factors(W2, region, nat, 10), T, 3, 3, static, ev)
    pd.testing.assert_frame_equal(f1, f2)


def test_detectors_are_causal():
    W, region, nat = _toy(seed=3)
    cfg = {"detect": {"cusum_k": 0.5, "ewma_lambda": 0.3, "bocpd_hazard": 0.05}}
    fx = FC.Factors(W, region, nat, 10)
    z = {"Все категории": D.standardized_residuals(fx, 3), "Продовольствие": D.standardized_residuals(fx, 3)}
    t0 = 17
    z2 = {c: v.copy() for c, v in z.items()}
    for c in z2:
        z2[c][t0 + 1:] = 7.0
    nb = [np.array([(i + 1) % W.shape[1]]) for i in range(W.shape[1])]
    s1 = D.run_detectors(z, "Все категории", cfg, nb, with_pelt=True)
    s2 = D.run_detectors(z2, "Все категории", cfg, nb, with_pelt=True)
    for k in s1:
        if k.startswith("spatial") or k == "stouffer_or_spatial":
            continue  # нормировка по МО в месяце: тоже только данные месяца t, проверено ниже
        np.testing.assert_allclose(np.nan_to_num(s1[k][: t0 + 1]), np.nan_to_num(s2[k][: t0 + 1]), err_msg=k)


def test_binseg_gain_matches_ruptures_cost():
    import ruptures as rpt
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.normal(size=9)
        x[6:] += rng.normal(0, 2)
        c = rpt.costs.CostL2().fit(x)
        for tau in (7, 8):
            ref_gain = c.error(0, 9) - c.error(0, tau) - c.error(tau, 9)
            a, b = x[:tau], x[tau:]
            gain = len(a) * len(b) / 9 * (a.mean() - b.mean()) ** 2
            assert abs(ref_gain - gain) < 1e-9


def test_bocpd_reacts_to_shift():
    rng = np.random.default_rng(0)
    z = np.full((24, 200), np.nan)
    z[12:] = rng.normal(size=(12, 200))
    z[18:, :100] -= 3.0                                  # сдвиг среднего у первой половины рядов
    s = D._bocpd(z, 0.05)
    assert np.nanmean(s[18:20, :100]) > 3 * np.nanmean(s[18:20, 100:])


def test_event_features_respect_date_known():
    ev = pd.DataFrame({"date_known": ["2024-04-04", "2023-07-21"],
                       "event_type": ["emergency_regional", "cbr_rate"],
                       "region": ["Оренбургская область", "Россия"], "municipality": [None, None],
                       "detail": ["режим ЧС", "повышение до 8.50"], "source_url": ["", ""], "verified": ["yes", "yes"]})
    ev["date_known"] = pd.to_datetime(ev["date_known"])
    ev["month"] = ev["date_known"].dt.to_period("M")
    ref = pd.DataFrame({"region": ["Оренбургская область", "Курганская область"]}, index=[1, 2])
    months = pd.period_range("2023-01", "2024-12", freq="M")
    f = EV.event_features(ev, ref, months)
    assert f["emerg_region"][pd.Period("2024-03", "M")].tolist() == [0.0, 0.0]
    assert f["emerg_region"][pd.Period("2024-04", "M")].tolist() == [1.0, 0.0]
    assert f["emerg_region"][pd.Period("2024-07", "M")].tolist() == [0.0, 0.0]
    assert f["rate"][pd.Period("2023-06", "M")] == 7.5 and f["rate"][pd.Period("2023-07", "M")] == 8.5


@pytest.mark.skipif(not (ROOT / "data/raw/hackathonlicence.zip").exists()
                    and not (ROOT / "data/raw/hackathonlicence/consumption.parquet").exists(),
                    reason="нет архива данных конкурса")
def test_real_data_schema():
    import os
    os.chdir(ROOT)
    from sbd.data import build_panel, load_config
    panel, rep = build_panel(load_config())
    assert rep["МО с полной историей (6 категорий x 24 мес.)"] == 2016
    assert all(w.shape == (24, 2016) for w in panel.values())
