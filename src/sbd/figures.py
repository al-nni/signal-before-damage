"""Графики для отчёта и презентации (PNG, 200 dpi). Все строятся из файлов outputs/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import matplotlib.dates  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# палитра: категориальные слоты по порядку, нейтральный серый для базовых линий
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GRAY, GRAY_L = "#8a8984", "#c9c8c2"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

MODEL_RU = {
    "prophet": "Prophet (по умолчанию)", "prophet_log_y": "Prophet (лог + годовая сезонность)",
    "naive": "Последнее значение", "snaive": "Сезонный наивный", "snaive_drift": "Сезонный наивный + рост г/г",
    "factor_national": "Фактор России (СберИндекс)", "factor_panel": "Общий фактор панели",
    "factor_region": "Фактор + регион", "lgbm": "Фактор + бустинг", "lgbm_no_events": "Бустинг без событий",
    "chronos_bolt": "Chronos-Bolt на отклонениях", "chronos_bolt_raw": "Chronos-Bolt на сыром ряду",
    "ensemble": "Ансамбль (фактор + свой сезон)", "ensemble_fm": "Ансамбль + Chronos-Bolt",
}
DET_RU = {
    "z_total": "z-оценка (все категории)", "cusum_total": "CUSUM (все категории)",
    "ewma_total": "EWMA (все категории)", "bocpd_total": "BOCPD (байесовский)",
    "binseg_total": "BinSeg (1 разрыв)", "pelt_total": "PELT (ruptures)",
    "stouffer": "Стауффер по 6 категориям", "stouffer_cusum": "CUSUM по Стауфферу",
    "spatial_stouffer": "Стауффер по соседям", "stouffer_or_spatial": "Стауффер: МО или соседи",
}


def _setup():
    for p in Path(__file__).resolve().parents[2].glob("assets/fonts/*.ttf"):
        font_manager.fontManager.addfont(str(p))
    fam = "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": fam, "font.size": 9.5, "axes.edgecolor": GRAY_L, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
        "legend.frameon": False, "axes.titleweight": "semibold", "axes.titlesize": 10.5,
        "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.axisbelow": True,
    })


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)


def fig_mae_by_h(summ: pd.DataFrame, total: str, out: Path):
    s = summ[(summ.category == total) & summ.h.isin([1, 3, 6, 12])]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    main = "ensemble_fm" if "ensemble_fm" in set(s.model) else "ensemble"
    spec = [("prophet", GRAY, "-", "o"), ("factor_region", ORANGE, "-", "D")]
    spec += [("ensemble", AQUA, "-", "s"), ("ensemble_fm", BLUE, "-", "o")] if main == "ensemble_fm" \
        else [("snaive_drift", AQUA, "-", "s"), ("ensemble", BLUE, "-", "o")]
    for m, col, ls, mk in spec:
        g = s[s.model == m].sort_values("h")
        if g.empty:
            continue
        ax.plot(g.h, g.MAE, color=col, ls=ls, lw=2, marker=mk, ms=6, label=MODEL_RU[m])
        if m in ("prophet", main):
            for h, v in zip(g.h, g.MAE):
                ax.annotate(f"{v:,.0f}".replace(",", " "), (h, v), xytext=(0, 8 if m == "prophet" else -14),
                            textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xticks([1, 3, 6, 12])
    ax.set_xlabel("Горизонт прогноза, мес.")
    ax.set_ylabel("MAE, руб. на жителя в месяц")
    ax.set_ylim(0, s[s.model.isin([m for m, *_ in spec])].MAE.max() * 1.35)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title("Ошибка прогноза трат «Все категории», 2 016 МО")
    _save(fig, out, "fig_mae_by_h.png")


def fig_gain_by_category(summ: pd.DataFrame, model: str, cats, out: Path):
    s = summ[(summ.model == model) & summ.h.isin([1, 3, 6])]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    y = np.arange(len(cats))
    w = 0.26
    for k, (h, col) in enumerate([(1, BLUE), (3, ORANGE), (6, AQUA)]):
        v = [-s[(s.category == c) & (s.h == h)]["MAE_vs_base_%"].iloc[0] for c in cats]
        ax.barh(y + (k - 1) * w, v, height=w - 0.03, color=col, label=f"h = {h} мес.")
        for yy, vv in zip(y + (k - 1) * w, v):
            ax.text(max(vv, 0) + 0.8, yy, f"{vv:.0f}%".replace("-", "−"), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    ax.set_xlabel("Снижение MAE относительно Prophet, % (меньше нуля — Prophet точнее)")
    lo = min(0.0, -s["MAE_vs_base_%"].max())
    ax.set_xlim(lo - 8 if lo < 0 else 0, 100)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{MODEL_RU[model]}: выигрыш у Prophet по категориям")
    _save(fig, out, "fig_gain_by_category.png")


def fig_detect_curve(rep: pd.DataFrame, main: str, out: Path):
    fars = [0.5, 1, 2, 3, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    means = rep.groupby("detector").mean(numeric_only=True)
    highlight = [main, "cusum_total", "pelt_total"]
    highlight = list(dict.fromkeys([d for d in highlight if d in means.index]))
    cols = [BLUE, ORANGE, AQUA]
    key = lambda f: f"recall@{int(f)}"                                # 0.5% хранится как recall@0
    for d in means.index:
        if d in highlight:
            continue
        ax.plot(fars, [means.loc[d, key(f)] for f in fars], color=GRAY_L, lw=1)
    for d, col in zip(highlight, cols):
        ax.plot(fars, [means.loc[d, key(f)] for f in fars], color=col, lw=2, marker="o", ms=5, label=DET_RU[d])
    ax.set_xscale("log")
    ax.set_xticks(fars)
    ax.set_xticklabels([f"{f:g}%" for f in fars])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Бюджет ложных тревог (доля МО-месяцев без шока)")
    ax.set_ylabel("Доля пойманных шоков за 3 мес.")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Полнота при фиксированном бюджете тревог (серые — прочие детекторы)")
    _save(fig, out, "fig_detect_curve.png")


def fig_recall_by_size(rep: pd.DataFrame, main: str, out: Path):
    means = rep[rep.detector == main].mean(numeric_only=True)
    sizes = [-20, -10, -5, 5, 10, 20]
    v = [means[f"recall@3_size{s}"] for s in sizes]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.bar(range(len(sizes)), v, color=[RED if s < 0 else BLUE for s in sizes], width=0.62)
    for k, vv in enumerate(v):
        ax.text(k, vv + 0.02, f"{vv:.0%}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s:+d}%" for s in sizes])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Доля пойманных")
    ax.set_xlabel("Величина шока (красное — провал, синее — всплеск)")
    ax.set_title(f"{DET_RU[main]}: полнота по величине шока при 3% ложных тревог")
    _save(fig, out, "fig_recall_by_size.png")


def fig_cost(rep: pd.DataFrame, main: str, n_mo: int, n_months: int, share: float, out: Path):
    """Стоимость ошибок при разной цене пропуска: выбор рабочей точки."""
    fars = [1, 2, 3, 5, 10, 20]
    g = rep[rep.detector == main].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    n_shock = share * n_mo
    for cm, col in [(5, AQUA), (10, BLUE), (20, ORANGE)]:
        cost = [(cm * (1 - g[f"recall@{f}"]) * n_shock + g[f"real_FAR@{f}"] * n_months * (n_mo - n_shock))
                / n_mo * 1000 for f in fars]
        ax.plot(fars, cost, color=col, lw=2, marker="o", ms=5, label=f"пропуск = {cm} ложных тревог")
        k = int(np.argmin(cost))
        ax.scatter([fars[k]], [cost[k]], s=90, facecolor="none", edgecolor=col, lw=1.5)
    ax.set_xscale("log")
    ax.set_xticks(fars)
    ax.set_xticklabels([f"{f}%" for f in fars])
    ax.set_xlabel("Бюджет ложных тревог")
    ax.set_ylabel("Потери на 1 000 МО, усл. ед.")
    ax.legend(fontsize=8)
    ax.set_title("Рабочая точка зависит от цены пропуска (кружок — минимум)")
    _save(fig, out, "fig_cost.png")


def fig_flood_lines(bt: pd.DataFrame, ref: pd.DataFrame, out: Path, names=("Орск", "Звериноголовский")):
    fig, axes = plt.subplots(1, len(names), figsize=(6.6, 2.9), sharey=False)
    for ax, nm in zip(np.atleast_1d(axes), names):
        tid = int(ref[ref.short == nm].index[0])
        g = bt[(bt.territory_id == tid) & (bt.category == "Все категории") & (bt.h == 1)]
        f = g[g.model == "factor_region"].sort_values("target")
        x = pd.PeriodIndex(f.target, freq="M").to_timestamp()
        ax.plot(x, f.y / 1000, color=INK, lw=2, label="факт")
        ax.plot(x, f.yhat / 1000, color=BLUE, lw=2, ls="--", label="прогноз на 1 мес.")
        ax.axvspan(pd.Timestamp("2024-03-17"), pd.Timestamp("2024-05-16"), color="#f0efec", zorder=0)
        ax.set_title(f"{nm} ({ref.loc[tid, 'region']})", fontsize=9)
        ax.tick_params(axis="x", labelrotation=0, labelsize=7.5)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m.%y"))
        ax.set_ylabel("тыс. руб. на жителя")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Паводок 2024 (серая полоса — апрель–май): траты против прогноза", x=0.01, ha="left",
                 fontsize=10.5, fontweight="semibold", color=INK)
    _save(fig, out, "fig_flood_lines.png")


def fig_flood_categories(rt: pd.DataFrame, out: Path):
    g = rt[rt.группа == "Оренбургская обл.: МО из постановлений о ЧС"]
    cats = ["Все категории", "Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(cats))
    for k, (m, col) in enumerate([("2024-04", ORANGE), ("2024-05", BLUE)]):
        v = [g[(g.категория == c) & (g.месяц == m)]["отклонение_%"].iloc[0] for c in cats]
        ax.bar(x + (k - 0.5) * 0.36, v, width=0.34, color=col, label={"2024-04": "апрель", "2024-05": "май"}[m])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("Общественное питание", "Общепит") for c in cats], fontsize=8)
    ax.set_ylabel("Факт к прогнозу, %")
    ax.legend(fontsize=8)
    ax.set_title(f"{int(g['МО'].iloc[0])} МО Оренбургской обл. с режимом ЧС: отклонение трат от прогноза")
    _save(fig, out, "fig_flood_categories.png")


def make_all(ctx):
    _setup()
    out = ctx.out / "figures"
    out.mkdir(exist_ok=True)
    summ = pd.read_csv(ctx.out / "table_forecast.csv")
    best = json.load(open(ctx.out / "best_model.json"))["best"] if (ctx.out / "best_model.json").exists() else "ensemble"
    fig_mae_by_h(summ, ctx.total, out)
    fig_gain_by_category(summ, best, ctx.cats, out)
    rep = pd.read_csv(ctx.out / "detect_by_repeat.csv")
    main = json.load(open(ctx.out / "cases" / "main_detector.json"))["main_detector"]
    fig_detect_curve(rep, main, out)
    fig_recall_by_size(rep, main, out)
    n_months = len(ctx.dates) - ctx.dates.get_loc(pd.Period(ctx.cfg["detect"]["eval_from"], "M"))
    fig_cost(rep, main, len(ctx.ids), n_months, ctx.cfg["synth"]["share_shocked"], out)
    bt = pd.read_parquet(ctx.out / "backtest.parquet", filters=[("h", "==", 1)])
    fig_flood_lines(bt, ctx.ref, out)
    fig_flood_categories(pd.read_csv(ctx.out / "cases" / "flood_residuals.csv"), out)
    print("графики:", sorted(p.name for p in out.glob("*.png")))
