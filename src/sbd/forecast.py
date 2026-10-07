"""Прогноз: «общий фактор месяца + отклонение МО» и его уточнения.

Обозначения (логарифмы трат на жителя):
    L[t, i]  — лог-траты МО i в месяце t (t = 0 … 23 соответствует 2023-01 … 2024-12);
    F[t]     — общий фактор месяца: медиана по МО отклонения L[t, i] от среднего МО за 2023 год;
    R[t, r]  — такой же фактор внутри региона r;
    N[t]     — лог ряда СберИндекса «расходы по России» (есть с 2018-12).
Все прогнозы из точки T используют только месяцы <= T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- факторы
class Factors:
    """Факторы по одной категории. Всё считается по данным <= T, поэтому
    при прогнозе из T используются только значения с индексами <= T."""

    def __init__(self, W: pd.DataFrame, region: pd.Series, nat: pd.Series, shrink_k: float):
        self.dates = W.index
        self.ids = np.asarray(W.columns)
        self.L = np.log(W.values)                                   # T x N
        self.abar = self.L[:12].mean(axis=0)                        # средний уровень МО за 2023
        dev0 = self.L - self.abar
        self.F = np.median(dev0, axis=1)                            # общий фактор
        reg = region.reindex(self.ids).values
        self.reg_codes, self.reg_idx = np.unique(reg, return_inverse=True)
        R = np.zeros((len(self.dates), len(self.reg_codes)))
        n_r = np.zeros(len(self.reg_codes))
        for r in range(len(self.reg_codes)):
            m = self.reg_idx == r
            R[:, r] = np.median(dev0[:, m], axis=1)
            n_r[r] = m.sum()
        self.R = R
        self.w_reg = (n_r / (n_r + shrink_k))[self.reg_idx]         # вес региональной сезонности
        # национальный ряд, выровненный так, что nat_at(t) доступен и для t < 0 (до 2023-01)
        self.nat = np.log(nat)
        self.nat_index = {p: k for k, p in enumerate(nat.index)}

    def nat_at(self, t: int) -> float:
        p = self.dates[0] + t                                       # t может быть отрицательным
        return self.nat.iloc[self.nat_index[p]]

    def dev(self) -> np.ndarray:
        """Отклонение МО от общего фактора: L - F (T x N)."""
        return self.L - self.F[:, None]

    def steps(self, T: int, h: int):
        """Сезонный шаг лог-уровня от T к T+h: общий, региональный, собственный МО, по России."""
        step_n = self.nat_at(T + h - 12) - self.nat_at(T - 12)
        if T - 12 >= 0:
            step_f = self.F[T + h - 12] - self.F[T - 12]
            step_r = (self.R[T + h - 12] - self.R[T - 12])[self.reg_idx]
            step_own = self.L[T + h - 12] - self.L[T - 12]
        else:  # в первой точке прогноза (2023-12) прошлогодних значений в панели нет
            step_f = step_n
            step_r = np.full(len(self.ids), step_n)
            step_own = np.full(len(self.ids), np.nan)
        return step_f, step_r, step_own, step_n


# ----------------------------------------------------------------------------- модели
def base_level(fx: Factors, T: int, m: int) -> np.ndarray:
    """Уровень МО в T: среднее отклонение от фактора за последние m месяцев + фактор в T."""
    d = fx.dev()[T - m + 1: T + 1].mean(axis=0)
    return d + fx.F[T]


def simple_models(fx: Factors, T: int, h: int, m: int) -> dict[str, np.ndarray]:
    """Прогнозы лог-уровня в T+h. Возвращает словарь модель -> вектор по МО."""
    L = fx.L
    step_f, step_r, step_own, step_n = fx.steps(T, h)
    yoy = (L[T] - L[T - 12]) if T - 12 >= 0 else np.full(L.shape[1], fx.nat_at(T) - fx.nat_at(T - 12))
    base = base_level(fx, T, m)
    w = fx.w_reg
    return {
        "naive": L[T].copy(),
        "snaive": L[T + h - 12].copy(),
        "snaive_drift": L[T + h - 12] + yoy,
        "factor_national": base + step_n,
        "factor_panel": base + step_f,
        "factor_region": base + w * step_r + (1 - w) * step_f,
    }


# ----------------------------------------------------------------------------- признаки для бустинга
def features(fx: Factors, T: int, h: int, m: int, static: pd.DataFrame, ev: dict) -> pd.DataFrame:
    """Признаки МО в точке прогноза T (все известны к концу месяца T)."""
    L, F = fx.L, fx.F
    D = fx.dev()
    step_f, step_r, step_own, step_n = fx.steps(T, h)
    n = L.shape[1]
    f = {}
    for k in (1, 2, 3):
        f[f"d_lag{k}"] = (D[T - k] - D[T]) if T - k >= 0 else np.full(n, np.nan)
    f["own_step_dev"] = step_own - step_f
    f["reg_step_dev"] = step_r - step_f
    f["nat_step_dev"] = np.full(n, step_n - step_f)
    f["own_yoy_dev"] = ((L[T] - L[T - 12]) - (F[T] - F[T - 12])) if T - 12 >= 0 else np.full(n, np.nan)
    tl = T + h - 12                                                  # тот же месяц год назад
    hist = D[: T + 1]
    f["own_season_dev"] = (D[tl] - hist.mean(axis=0)) if 0 <= tl <= T else np.full(n, np.nan)
    f["vol"] = np.std(np.diff(hist, axis=0), axis=0) if T >= 2 else np.full(n, np.nan)
    f["target_month"] = np.full(n, (fx.dates[0] + T + h).month)
    f["h"] = np.full(n, h)
    for c in static.columns:
        f[c] = static[c].values
    # событийный слой: изменение ключевой ставки за 3 мес. до T и действующий режим ЧС в регионе в T
    f["key_rate_chg_3m"] = np.full(n, ev["rate_chg_3m"].get(fx.dates[T], 0.0))
    f["emergency_region"] = ev["emerg_region"].get(fx.dates[T], np.zeros(n))
    return pd.DataFrame(f)


def static_features(ref: pd.DataFrame, market_access: pd.Series) -> pd.DataFrame:
    s = pd.DataFrame(index=ref.index)
    s["log_pop"] = np.log(ref.population.fillna(ref.population.median()).clip(lower=100))
    s["log_wage"] = np.log(ref.wage.fillna(ref.wage.median()))
    s["market_access"] = market_access.reindex(ref.index).values
    s["lat"] = ref.lat.values
    s["lon"] = ref.lon.values
    s["mo_type"] = pd.Categorical(ref.mo_type).codes
    return s


# ----------------------------------------------------------------------------- бэктест
def backtest(panel: dict, ref: pd.DataFrame, nat: pd.DataFrame, static: pd.DataFrame,
             ev_by_cat: dict, cfg: dict, use_lgbm: bool = True, lgbm_event_features: bool = True,
             simple: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Прогнозы всех моделей из всех точек T на все горизонты до конца 2024 года.

    Возвращает длинную таблицу: category, territory_id, origin, h, target, model, yhat, y.
    """
    import lightgbm as lgb

    fc = cfg["forecast"]
    m = fc["level_months"]
    cats = cfg["data"]["categories"]
    fxs = {c: Factors(panel[c], ref.region, nat[cfg["data"]["national_map"][c]],
                      fc["region_shrink_k"]) for c in cats}
    dates = panel[cats[0]].index
    ids = np.asarray(panel[cats[0]].columns)
    t_from = dates.get_loc(pd.Period(fc["origins_from"], "M"))
    t_to = dates.get_loc(pd.Period(fc["origins_to"], "M"))
    last = len(dates) - 1
    rows = []

    # 1) простые модели
    for c in (cats if simple else []):
        fx = fxs[c]
        for T in range(t_from, t_to + 1):
            for h in range(1, last - T + 1):
                for name, lp in simple_models(fx, T, h, m).items():
                    rows.append(pd.DataFrame({"category": c, "territory_id": ids, "origin": str(dates[T]),
                                              "h": h, "target": str(dates[T + h]), "model": name,
                                              "yhat": np.exp(lp), "y": np.exp(fx.L[T + h])}))

    # 2) бустинг: поправка к factor_region, общий для всех категорий, отдельная модель на (T, h)
    if use_lgbm:
        # n_jobs=1: при параллельно работающем Prophet многопоточность LightGBM резко замедляется
        params = dict(fc["lgbm"], random_state=cfg["seed"], verbose=-1, n_jobs=1)
        drop = [] if lgbm_event_features else ["key_rate_chg_3m", "emergency_region"]
        for T in range(t_from, t_to + 1):
            for h in [h for h in fc["horizons_eval"] if h <= last - T]:
                Xtr, ytr = [], []
                # обучаем только на точках s >= 2024-01 (есть прошлогодние значения): до этого базовая
                # модель берёт сезонность из ряда по России, и поправки из того режима переносить нельзя
                for s in range(12, T - h + 1):                       # цели известны к T: s + h <= T
                    for ci, c in enumerate(cats):
                        fx = fxs[c]
                        X = features(fx, s, h, m, static, ev_by_cat[c]).drop(columns=drop)
                        X["cat"] = ci
                        base = simple_models(fx, s, h, m)["factor_region"]
                        Xtr.append(X)
                        ytr.append(fx.L[s + h] - base)
                preds = {}
                for ci, c in enumerate(cats):
                    fx = fxs[c]
                    base = simple_models(fx, T, h, m)["factor_region"]
                    if Xtr:
                        Xq = features(fx, T, h, m, static, ev_by_cat[c]).drop(columns=drop)
                        Xq["cat"] = ci
                        preds[c] = (base, Xq)
                    else:
                        preds[c] = (base, None)
                if Xtr:
                    X_all = pd.concat(Xtr, ignore_index=True)
                    y_all = np.concatenate(ytr)
                    model = lgb.LGBMRegressor(objective="l1", **params)
                    model.fit(X_all, y_all, categorical_feature=["cat", "mo_type"])
                for c in cats:
                    base, Xq = preds[c]
                    lp = base + (model.predict(Xq) if Xq is not None else 0.0)
                    rows.append(pd.DataFrame({"category": c, "territory_id": ids, "origin": str(dates[T]),
                                              "h": h, "target": str(dates[T + h]),
                                              "model": "lgbm" if lgbm_event_features else "lgbm_no_events",
                                              "yhat": np.exp(lp), "y": np.exp(fxs[c].L[T + h])}))
                if verbose:
                    print(f"  lgbm T={dates[T]} h={h} train={0 if not Xtr else len(y_all)}", flush=True)
    return pd.concat(rows, ignore_index=True)


def add_ensemble(bt: pd.DataFrame, members=("factor_region", "snaive_drift"), name="ensemble") -> pd.DataFrame:
    """Геометрическое среднее участников (веса равные, заданы заранее, не подбираются на тесте)."""
    key = ["category", "territory_id", "origin", "h", "target"]
    sub = bt[bt.model.isin(members)]
    g = sub.assign(ly=np.log(sub.yhat)).groupby(key).agg(ly=("ly", "mean"), y=("y", "first"),
                                                        n=("ly", "size")).reset_index()
    g = g[g.n == len(members)]
    g["yhat"] = np.exp(g.ly)
    g["model"] = name
    return pd.concat([bt, g[key + ["model", "yhat", "y"]]], ignore_index=True)
