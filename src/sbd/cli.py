"""Точка входа: python -m sbd.cli {validate|forecast|detect|cases|figures|all}.

Все результаты пишутся в outputs/. Отчёт и презентация собираются из этих файлов
скриптом report/build.py, поэтому числа в них всегда совпадают с расчётом.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cases as CS
from . import detect as D
from . import events as EV
from . import forecast as FC
from . import foundation as FD
from . import metrics as M
from . import prophet_baseline as PB
from .data import (build_panel, load_config, load_national, load_reference, neighbours,
                   validate_national, validate_reference)


class Ctx:
    """Общие для всех шагов объекты (данные, справочник, соседи, события)."""

    def __init__(self):
        self.cfg = load_config()
        self.out = Path(self.cfg["paths"]["out"])
        self.panel, self.report = build_panel(self.cfg)
        self.cats = self.cfg["data"]["categories"]
        self.total = self.cfg["data"]["total_category"]
        self.dates = self.panel[self.total].index
        self.ids = np.asarray(self.panel[self.total].columns)
        self.pos = {v: k for k, v in enumerate(self.ids)}
        self.ref = load_reference(self.cfg, self.ids)
        self.nat = load_national(self.cfg)
        self.events = EV.load_events(self.cfg["paths"]["events"])
        self._nb = None

    @property
    def nb_index(self):
        if self._nb is None:
            nb = neighbours(self.cfg, self.ids, self.cfg["detect"]["neighbours_km"])
            self._nb = [np.array([self.pos[j] for j in nb[int(i)]], dtype=int) for i in self.ids]
        return self._nb

    def factors(self, panel=None):
        panel = panel or self.panel
        return {c: FC.Factors(panel[c], self.ref.region, self.nat[self.cfg["data"]["national_map"][c]],
                              self.cfg["forecast"]["region_shrink_k"]) for c in self.cats}


# ----------------------------------------------------------------------------- 1. данные
def step_validate(ctx: Ctx):
    rep = {k: (v.item() if hasattr(v, "item") else v) for k, v in ctx.report.items()}
    rep["справочник: проверка по автодорожным расстояниям"] = validate_reference(ctx.cfg, ctx.ref)
    rep["ряд СберИндекса по России: сверка с панелью"] = validate_national(ctx.panel, ctx.ref, ctx.nat, ctx.cfg)
    mm = EV.match_municipalities(ctx.events, ctx.ref)
    mm.to_csv(ctx.out / "event_matching.csv", index=False)
    rep["события: МО из сообщений найдены в панели"] = f"{int(mm.in_panel.sum())} из {len(mm)}"
    (ctx.out / "data_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps(rep, ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------------- 2. прогноз
def step_forecast(ctx: Ctx):
    cfg = ctx.cfg
    mk = pd.read_parquet(Path(cfg["paths"]["raw_dir"]) / "market_access.parquet").set_index("territory_id")
    static = FC.static_features(ctx.ref, mk.market_access)
    evf = EV.event_features(ctx.events, ctx.ref, ctx.dates)
    ev_by_cat = {c: evf for c in ctx.cats}
    cache = ctx.out / "cache"
    cache.mkdir(exist_ok=True)
    f_bt, f_ne = cache / "bt_models.parquet", cache / "bt_lgbm_no_events.parquet"
    if f_bt.exists():
        bt = pd.read_parquet(f_bt)
    else:
        t = time.time()
        bt = FC.backtest(ctx.panel, ctx.ref, ctx.nat, static, ev_by_cat, cfg, use_lgbm=True, verbose=False)
        bt.to_parquet(f_bt)
        print(f"простые модели + бустинг: {time.time() - t:.0f} с")
    if f_ne.exists():
        bt_ne = pd.read_parquet(f_ne)
    else:
        bt_ne = FC.backtest(ctx.panel, ctx.ref, ctx.nat, static, ev_by_cat, cfg, use_lgbm=True,
                            lgbm_event_features=False, simple=False, verbose=False)
        bt_ne.to_parquet(f_ne)
        print("бустинг без событийных признаков — готово")
    pr = PB.run(ctx.panel, cfg, "prophet", ctx.cats)
    pr2 = PB.run(ctx.panel, cfg, "prophet_log_y", [ctx.total])
    parts = [bt, bt_ne]
    for p in (pr, pr2):
        p = p.merge(bt[bt.model == "naive"][["category", "territory_id", "origin", "h", "y"]],
                    on=["category", "territory_id", "origin", "h"])
        parts.append(p[["category", "territory_id", "origin", "h", "target", "model", "yhat", "y"]])
    chronos_dir = cfg["forecast"].get("chronos_dir", "models/chronos-bolt-small")
    have_chronos = FD.available(chronos_dir)
    if have_chronos:
        sha = FD.weights_sha256(chronos_dir)
        if sha != FD.SHA256_SMALL:
            print(f"ВНИМАНИЕ: SHA-256 весов {sha} не совпадает с опубликованным для chronos-bolt-small")
        f_ch = cache / "bt_chronos.parquet"
        if f_ch.exists():
            ch = pd.read_parquet(f_ch)
        else:
            t = time.time()
            ch = FD.backtest_chronos(ctx.panel, ctx.ref, ctx.nat, cfg, chronos_dir)
            ch.to_parquet(f_ch)
            print(f"Chronos-Bolt: {time.time() - t:.0f} с")
        parts.append(ch)
        json.dump({"model": "amazon/chronos-bolt-small", "sha256": sha, "matches_published": sha == FD.SHA256_SMALL},
                  open(ctx.out / "foundation_model.json", "w"), indent=1)
    else:
        print(f"Chronos-Bolt: веса не найдены в {chronos_dir} — модель пропущена")
    allbt = pd.concat(parts, ignore_index=True)
    # ансамбли заданы заранее (равные веса, без подбора на тесте):
    #   ensemble    — «коллективный» прогноз (фактор + регион) и «индивидуальный» (свой сезон + рост г/г);
    #   ensemble_fm — те же два плюс фундаментальная модель Chronos-Bolt на отклонениях от фактора.
    allbt = FC.add_ensemble(allbt, members=("factor_region", "snaive_drift"), name="ensemble")
    if have_chronos:
        allbt = FC.add_ensemble(allbt, members=("factor_region", "snaive_drift", "chronos_bolt"),
                                name="ensemble_fm")
    allbt["territory_id"] = allbt.territory_id.astype(int)
    allbt.to_parquet(ctx.out / "backtest.parquet")
    summ = M.summarize(allbt, baseline="prophet")
    summ.to_csv(ctx.out / "table_forecast.csv", index=False)
    # лучшая модель: наименьшая средняя MAE по всем категориям и горизонтам 1, 3, 6, 12
    cand = summ[summ.h.isin([1, 3, 6, 12]) & ~summ.model.isin(["prophet", "prophet_log_y"])]
    rank = cand.groupby("model").MAE.mean().sort_values()
    json.dump({"best": rank.index[0], "ranking": rank.round(1).to_dict()},
              open(ctx.out / "best_model.json", "w"), ensure_ascii=False, indent=1)
    print("лучшая модель:", rank.index[0])
    # сравнение с лучшим из двух вариантов Prophet по «Все категории»
    s2 = M.summarize(allbt[allbt.category == ctx.total], baseline="prophet_log_y")
    s2.to_csv(ctx.out / "table_forecast_vs_prophet_log_y.csv", index=False)
    print(summ[(summ.category == ctx.total) & summ.h.isin([1, 3, 6, 12])]
          [["h", "model", "MAE", "WAPE", "R2", "MAE_vs_base_%", "share_MO_better", "DM_p"]].round(3).to_string())


# ----------------------------------------------------------------------------- 3. детекция
def _scores(ctx: Ctx, panel, with_pelt=True):
    fxs = ctx.factors(panel)
    z = {c: D.standardized_residuals(fxs[c], ctx.cfg["forecast"]["level_months"]) for c in ctx.cats}
    return D.run_detectors(z, ctx.total, ctx.cfg, ctx.nb_index, with_pelt=with_pelt), z


CURVE_FAR = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]


def step_detect(ctx: Ctx):
    cfg = ctx.cfg
    t_from = ctx.dates.get_loc(pd.Period(cfg["detect"]["eval_from"], "M"))
    clean, z = _scores(ctx, ctx.panel)
    np.savez_compressed(ctx.out / "scores_clean.npz", **{k: v for k, v in clean.items()},
                        **{f"z__{c}": v for c, v in z.items()})
    th = D.thresholds(clean, t_from, sorted(set(CURVE_FAR + cfg["budget"]["far_levels"])))
    json.dump(th, open(ctx.out / "thresholds.json", "w"), ensure_ascii=False, indent=1)
    rng = np.random.default_rng(cfg["seed"])
    res, curves, metas = [], [], []
    cfg_curve = dict(cfg, budget=dict(cfg["budget"], far_levels=CURVE_FAR))
    for rep in range(cfg["synth"]["repeats"]):
        t = time.time()
        P, meta = D.inject(ctx.panel, ctx.ids, ctx.nb_index, cfg, rng)
        sc, _ = _scores(ctx, P)
        r = D.evaluate(sc, meta, th, t_from, cfg_curve)
        r["repeat"] = rep
        res.append(r)
        metas.append(meta.assign(repeat=rep))
        print(f"  повтор {rep + 1}: {len(meta)} МО с шоком, {time.time() - t:.0f} с", flush=True)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(ctx.out / "detect_by_repeat.csv", index=False)
    pd.concat(metas).drop(columns=["onset"]).assign(onset=lambda d: d.onset_idx).to_csv(
        ctx.out / "synthetic_shocks.csv", index=False)
    agg = res.drop(columns="repeat").groupby("detector").agg(["mean", "std"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    agg.to_csv(ctx.out / "table_detect.csv")
    print(res.groupby("detector")[["PR_AUC", "precision@20", "recall@1", "recall@3", "recall@5",
                                   "real_FAR@3", "median_delay@3", "cost@3_per1000MO"]].mean().round(3)
          .sort_values("cost@3_per1000MO").to_string())


# ----------------------------------------------------------------------------- 4. кейсы и новости
def step_cases(ctx: Ctx):
    cfg = ctx.cfg
    out = ctx.out / "cases"
    out.mkdir(exist_ok=True)
    t_from = ctx.dates.get_loc(pd.Period(cfg["detect"]["eval_from"], "M"))
    npz = np.load(ctx.out / "scores_clean.npz")
    th = json.load(open(ctx.out / "thresholds.json"))
    det = pd.read_csv(ctx.out / "table_detect.csv", index_col=0)
    main = det["cost@3_per1000MO__mean"].idxmin()                    # выбран на полусинтетике
    z = {c: npz[f"z__{c}"] for c in ctx.cats}
    fxs = ctx.factors()
    res_share = {c: D.residual_share(fxs[c], cfg["forecast"]["level_months"]) for c in ctx.cats}
    mm = pd.read_csv(ctx.out / "event_matching.csv")
    mm = mm[mm.in_panel]
    oren = mm[(mm.region == "Оренбургская область")].territory_id.astype(int).values
    kt = mm[mm.region.isin(["Курганская область", "Тюменская область"])].territory_id.astype(int).values
    reg_oren = ctx.ref[ctx.ref.region == "Оренбургская область"].index.values
    reg_kt = ctx.ref[ctx.ref.region.isin(["Курганская область", "Тюменская область"])].index.values
    idx = lambda arr: np.array([ctx.pos[i] for i in arr], dtype=int)
    others = np.setdiff1d(np.arange(len(ctx.ids)), idx(np.concatenate([reg_oren, reg_kt])))
    groups = {
        "Оренбургская обл.: МО из постановлений о ЧС": idx(oren),
        "Оренбургская обл.: остальные МО": idx(np.setdiff1d(reg_oren, oren)),
        "Курганская и Тюменская обл.: названные МО": idx(kt),
        "Курганская и Тюменская обл.: остальные МО": idx(np.setdiff1d(reg_kt, kt)),
        "Остальная Россия": others,
    }
    months = ["2024-03", "2024-04", "2024-05", "2024-06", "2024-07"]
    rt = CS.residual_table(res_share, ctx.ids, groups, ctx.dates, months)
    rt.to_csv(out / "flood_residuals.csv", index=False)
    # отдельные МО для иллюстрации
    ill = []
    for tid in list(oren) + list(kt):
        i = ctx.pos[tid]
        for c in ctx.cats:
            for p in months:
                t = ctx.dates.get_loc(pd.Period(p, "M"))
                ill.append({"territory_id": tid, "МО": ctx.ref.loc[tid, "short"], "регион": ctx.ref.loc[tid, "region"],
                            "категория": c, "месяц": p, "отклонение_%": 100 * res_share[c][t, i],
                            "z": z[c][t, i]})
    pd.DataFrame(ill).to_csv(out / "flood_mo_detail.csv", index=False)
    # тревоги и плацебо
    rng = np.random.default_rng(cfg["seed"])
    t_apr = ctx.dates.get_loc(pd.Period("2024-04", "M"))
    rows = []
    named_all = idx(np.concatenate([oren, kt]))
    for detname in [main, "stouffer", "cusum_total", "z_total", "stouffer_cusum", "pelt_total"]:
        if detname not in npz.files:
            continue
        S = np.nan_to_num(npz[detname], nan=-np.inf)
        for far in ["0.03", "0.05"]:
            for gname, g in [("все названные МО", named_all), ("Оренбургская обл. (постановления)", idx(oren)),
                             ("Курганская и Тюменская обл.", idx(kt))]:
                r = CS.alarm_test(S, th[detname][far], g, t_apr, 3, (t_from, len(ctx.dates) - 1),
                                  cfg["case"]["placebo_draws"], rng)
                r.update({"детектор": detname, "FAR": far, "группа": gname})
                rows.append(r)
    pd.DataFrame(rows).to_csv(out / "flood_alarm_test.csv", index=False)
    # «подсказка новостей»: снижение порога в регионах с действующим режимом ЧС
    evf = EV.event_features(ctx.events, ctx.ref, ctx.dates)
    emerg = np.vstack([evf["emerg_region"][p] for p in ctx.dates])
    S = np.nan_to_num(npz[main], nan=-np.inf)
    g_rows = []
    for fb, fg in [("0.03", "0.03"), ("0.03", "0.1"), ("0.03", "0.2"), ("0.01", "0.1"), ("0.01", "0.2")]:
        A = CS.guided_alarms(S, th[main][fb], th[main][fg], emerg)
        A[:t_from] = False
        base = S >= th[main][fb]
        base[:t_from] = False
        g_rows.append({"детектор": main, "FAR вне зоны ЧС": fb, "FAR в зоне ЧС": fg,
                       "названных МО с тревогой (апр–июн)": int(A[t_apr:t_apr + 3, named_all].any(axis=0).sum()),
                       "из них без подсказки": int(base[t_apr:t_apr + 3, named_all].any(axis=0).sum()),
                       "названных МО всего": len(named_all),
                       "дополнительных тревог за год": int(A.sum() - base.sum()),
                       "всего тревог за год": int(A.sum())})
    pd.DataFrame(g_rows).to_csv(out / "news_guided.csv", index=False)
    # по каждому названному МО: сильнейший сигнал апреля–июня и тревоги при разных порогах
    pct = pd.DataFrame(S).rank(axis=1, pct=True).values
    st_signed = sum(z[c] for c in ctx.cats) / np.sqrt(len(ctx.cats))
    A_guided = CS.guided_alarms(S, th[main]["0.03"], th[main]["0.2"], emerg)
    mo_rows = []
    for tid in list(oren) + list(kt):
        i = ctx.pos[int(tid)]
        w = slice(t_apr, t_apr + 3)
        k = int(np.argmax(S[w, i]))
        mo_rows.append({"МО": ctx.ref.loc[tid, "short"], "регион": ctx.ref.loc[tid, "region"],
                        "месяц максимума": str(ctx.dates[t_apr + k]),
                        "направление": "всплеск" if st_signed[t_apr + k, i] > 0 else "провал",
                        "оценка": float(S[t_apr + k, i]), "перцентиль среди МО": float(pct[t_apr + k, i]),
                        "тревога при 3%": bool((S[w, i] >= th[main]["0.03"]).any()),
                        "тревога при 5%": bool((S[w, i] >= th[main]["0.05"]).any()),
                        "тревога с подсказкой новостей": bool(A_guided[w, i].any())})
    pd.DataFrame(mo_rows).to_csv(out / "flood_mo_alarms.csv", index=False)
    # список сильнейших тревог — очередь аналитика
    st_signed = sum(z[c] for c in ctx.cats) / np.sqrt(len(ctx.cats))
    ta = CS.top_alarms(npz["stouffer"], st_signed, z, ctx.ref, ctx.dates, t_from, k=25)
    ta.to_csv(out / "top_alarms.csv", index=False)
    json.dump({"main_detector": main}, open(out / "main_detector.json", "w"))
    print("основной детектор:", main)
    print(pd.DataFrame(rows).round(3).to_string())
    print(pd.DataFrame(g_rows).to_string())
    print(rt[rt.категория == ctx.total].pivot(index="группа", columns="месяц", values="отклонение_%").round(1))


# ----------------------------------------------------------------------------- 5. графики
def step_figures(ctx: Ctx):
    from . import figures
    figures.make_all(ctx)


STEPS = {"validate": step_validate, "forecast": step_forecast, "detect": step_detect,
         "cases": step_cases, "figures": step_figures}


def main(argv=None):
    argv = argv or sys.argv[1:]
    cmd = argv[0] if argv else "all"
    ctx = Ctx()
    todo = list(STEPS) if cmd == "all" else [cmd]
    for s in todo:
        print(f"=== {s}", flush=True)
        STEPS[s](ctx)


if __name__ == "__main__":
    main()
