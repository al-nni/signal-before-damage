"""Обнаружение шоков и оценка детекторов «как антифрод-систем».

Вход детекторов — стандартизированный остаток одношагового прогноза:
    r[t, i] = L[t, i] - прогноз(t-1 -> t) модели factor_region;
    r*[t, i] = r[t, i] - медиана_i r[t, i]      (убираем общий для страны промах месяца);
    z[t, i] = r*[t, i] / sigma_i,                sigma_i — устойчивый разброс МО по 2023 году.
Детекторы двусторонние: ловят и провал трат (ущерб, эвакуация), и всплеск (компенсации,
восстановление, ажиотаж) — для банка важны оба. Все детекторы онлайн: оценка в месяце t
использует только данные <= t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import Factors, simple_models

T0 = 12      # 2024-01: первый месяц, для которого есть одношаговый прогноз


# ----------------------------------------------------------------------------- остатки
def standardized_residuals(fx: Factors, m: int) -> np.ndarray:
    """z (24 x N), NaN до 2024-01."""
    L = fx.L
    n_t, n = L.shape
    r = np.full((n_t, n), np.nan)
    for t in range(T0, n_t):
        r[t] = L[t] - simple_models(fx, t - 1, 1, m)["factor_region"]
    r = r - np.nanmedian(r, axis=1, keepdims=True)
    d = np.diff(fx.dev()[:12], axis=0)                               # только 2023 год
    sig = 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0)
    sig = np.maximum(sig, np.quantile(sig, 0.05))
    return r / sig


def residual_share(fx: Factors, m: int) -> np.ndarray:
    """Отклонение факта от прогноза в долях (для интерпретации кейсов), 24 x N."""
    L = fx.L
    out = np.full(L.shape, np.nan)
    for t in range(T0, L.shape[0]):
        out[t] = np.exp(L[t] - simple_models(fx, t - 1, 1, m)["factor_region"]) - 1
    med = np.nanmedian(out, axis=1, keepdims=True)
    return out - med


# ----------------------------------------------------------------------------- детекторы
def _cusum(z: np.ndarray, k: float) -> np.ndarray:
    """Двусторонний CUSUM (Page, 1954): max(S+, S-)."""
    out = np.full(z.shape, np.nan)
    up = np.zeros(z.shape[1])
    dn = np.zeros(z.shape[1])
    for t in range(T0, z.shape[0]):
        x = np.nan_to_num(z[t])
        up = np.maximum(0.0, up + x - k)
        dn = np.maximum(0.0, dn - x - k)
        out[t] = np.maximum(up, dn)
    return out


def _ewma(z: np.ndarray, lam: float) -> np.ndarray:
    """|EWMA| стандартизированных остатков."""
    out = np.full(z.shape, np.nan)
    acc = np.zeros(z.shape[1])
    for t in range(T0, z.shape[0]):
        acc = lam * np.nan_to_num(z[t]) + (1 - lam) * acc
        out[t] = np.abs(acc)
    return out


def _bocpd(z: np.ndarray, hazard: float, prior_var: float = 1.0) -> np.ndarray:
    """Байесовская онлайн-детекция разладки (Adams & MacKay, 2007) для сдвига среднего
    нормального ряда с единичной дисперсией. Оценка = P(разладка в последние 2 месяца)
    с учётом величины сдвига: P * |среднее нового участка|."""
    n_t, n = z.shape
    out = np.full(z.shape, np.nan)

    def pdf(x, mean, var):
        return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)

    R = np.ones((n, 1))                     # P(длина текущего участка = r), до первого наблюдения
    s_sum = np.zeros((n, 1))                # сумма наблюдений участка для каждой длины r
    cnt = np.zeros(1)                       # число наблюдений участка для каждой длины r
    for t in range(T0, n_t):
        x = np.nan_to_num(z[t])[:, None]
        post_var = 1.0 / (1.0 / prior_var + cnt[None, :])
        pred = pdf(x, post_var * s_sum, post_var + 1.0)             # участок продолжается
        pred0 = pdf(x, 0.0, prior_var + 1.0)                        # x — первое наблюдение нового участка
        grow = R * pred * (1 - hazard)
        cp = R.sum(axis=1, keepdims=True) * hazard * pred0
        R = np.hstack([cp, grow])
        R /= R.sum(axis=1, keepdims=True)
        s_sum = np.hstack([np.zeros((n, 1)), s_sum]) + x
        cnt = np.concatenate([[0.0], cnt]) + 1.0
        # R[:, 0] — разладка перед текущим месяцем, R[:, 1] — перед предыдущим
        r0, r1 = R[:, 0], (R[:, 1] if R.shape[1] > 1 else 0.0)
        m0 = s_sum[:, 0]
        m1 = s_sum[:, 1] / 2.0 if s_sum.shape[1] > 1 else m0
        mean_new = (r0 * m0 + r1 * m1) / np.maximum(r0 + r1, 1e-12)
        out[t] = (r0 + r1) * np.abs(mean_new)
    return out


def _binseg_lr(z: np.ndarray, recent: int = 2) -> np.ndarray:
    """Одна точка разладки на расширяющемся окне (Binary Segmentation, стоимость l2):
    выигрыш от разбиения n1*n2/(n1+n2)*(m1-m2)^2 для разрезов в последних `recent` месяцах.
    Совпадает с ruptures.Binseg(model='l2') по выбору
    точки на коротком окне (проверяется в тестах)."""
    n_t, n = z.shape
    out = np.full(z.shape, np.nan)
    for t in range(T0, n_t):
        seg = np.nan_to_num(z[T0: t + 1])
        L = seg.shape[0]
        best = np.zeros(n)
        for tau in range(max(1, L - recent), L):                    # разрез перед позицией tau
            a, b = seg[:tau], seg[tau:]
            n1, n2 = len(a), len(b)
            m1, m2 = a.mean(axis=0), b.mean(axis=0)
            gain = n1 * n2 / (n1 + n2) * (m1 - m2) ** 2
            best = np.maximum(best, gain)
        out[t] = best
    return out


def _pelt(z: np.ndarray, penalties=(40, 25, 16, 10, 7, 5, 3.5, 2.5, 1.7, 1.2), recent: int = 2) -> np.ndarray:
    """PELT (ruptures) на расширяющемся окне. Непрерывная оценка = наибольший штраф из сетки,
    при котором PELT всё ещё находит разладку в последних `recent` месяцах."""
    import ruptures as rpt
    n_t, n = z.shape
    out = np.full(z.shape, np.nan)
    for t in range(T0, n_t):
        seg = np.nan_to_num(z[T0: t + 1])
        L = seg.shape[0]
        col = np.zeros(n)
        if L >= 3:
            for i in range(n):
                x = seg[:, i]
                for pen in penalties:
                    bk = rpt.Pelt(model="l2", min_size=1, jump=1).fit(x).predict(pen=pen)[:-1]
                    if bk and bk[-1] >= L - recent:
                        col[i] = pen
                        break
        out[t] = col
    return out


def run_detectors(z_by_cat: dict[str, np.ndarray], total: str, cfg: dict,
                  nb_index: list[np.ndarray] | None, with_pelt: bool = True) -> dict[str, np.ndarray]:
    d = cfg["detect"]
    zt = z_by_cat[total]
    cats = list(z_by_cat)
    st = sum(z_by_cat[c] for c in cats) / np.sqrt(len(cats))         # сумма Стауффера по категориям
    out = {
        "z_total": np.abs(zt),
        "cusum_total": _cusum(zt, d["cusum_k"]),
        "ewma_total": _ewma(zt, d["ewma_lambda"]),
        "bocpd_total": _bocpd(zt, d["bocpd_hazard"]),
        "binseg_total": _binseg_lr(zt),
        "stouffer": np.abs(st),
        "stouffer_cusum": _cusum(st, d["cusum_k"]),
    }
    if with_pelt:
        out["pelt_total"] = _pelt(zt)
    if nb_index is not None:
        # среднее по МО и его ближайшим соседям; нормируем на устойчивый разброс по МО в месяце,
        # чтобы шкала была сравнима с одиночной оценкой (у соседей шум коррелирован)
        pooled = np.full(st.shape, np.nan)
        for i, nb in enumerate(nb_index):
            idx = np.concatenate([[i], nb]).astype(int)
            pooled[:, i] = st[:, idx].mean(axis=1)
        med = np.nanmedian(pooled, axis=1, keepdims=True)
        mad = 1.4826 * np.nanmedian(np.abs(pooled - med), axis=1, keepdims=True)
        pooled = (pooled - med) / mad
        out["spatial_stouffer"] = np.abs(pooled)
        out["stouffer_or_spatial"] = np.fmax(np.abs(st), np.abs(pooled))
    return out


# ----------------------------------------------------------------------------- полусинтетика
CATEGORY_PROFILE = {   # доля шока по категориям относительно «Все категории» (допущение)
    "Все категории": 1.0, "Продовольствие": 0.5, "Здоровье": 0.5,
    "Маркетплейсы": 1.0, "Общественное питание": 1.5, "Транспорт": 1.5,
}


def inject(panel: dict, ids: np.ndarray, nb_index: list[np.ndarray], cfg: dict, rng) -> tuple[dict, pd.DataFrame]:
    """Вставляет шоки в реальные ряды. Часть шоков «пространственные»: задевают и соседей МО."""
    s = cfg["synth"]
    dates = panel[next(iter(panel))].index
    n = len(ids)
    n_target = int(s["share_shocked"] * n)                          # всего МО с шоком (вместе с соседями)
    taken = np.zeros(n, bool)
    meta = []
    seeds = rng.permutation(n)
    for i in seeds:
        if taken.sum() >= n_target:
            break
        if taken[i]:
            continue
        size = float(rng.choice(s["sizes"]))
        shape = str(rng.choice(s["shapes"]))
        onset = pd.Period(str(rng.choice(s["onsets"])), "M")
        cluster = rng.random() < s["cluster_share"]
        members = [i] + ([j for j in nb_index[i] if not taken[j]] if cluster else [])
        members = members[: max(1, n_target - int(taken.sum()))]
        for j in members:
            taken[j] = True
            meta.append({"idx": j, "territory_id": ids[j], "seed": ids[i], "size": size, "shape": shape,
                         "onset": onset, "cluster": cluster and len(members) > 1, "is_seed": j == i})
    meta = pd.DataFrame(meta)
    t_on = {r.idx: dates.get_loc(r.onset) for r in meta.itertuples()}
    out = {}
    for c, W in panel.items():
        X = W.values.copy()
        prof = CATEGORY_PROFILE.get(c, 1.0)
        for r in meta.itertuples():
            t0 = t_on[r.idx]
            mult = np.ones(len(dates))
            for t in range(t0, len(dates)):
                frac = 1.0 if r.shape == "step" else min(1.0, (t - t0 + 1) / s["ramp_len"])
                mult[t] = 1.0 + prof * r.size * frac
            X[:, r.idx] *= mult
        out[c] = pd.DataFrame(X, index=W.index, columns=W.columns)
    meta["onset_idx"] = [t_on[i] for i in meta.idx]
    return out, meta


# ----------------------------------------------------------------------------- бюджет тревог
def thresholds(scores: dict[str, np.ndarray], t_from: int, far_levels) -> dict:
    """Порог под заданную долю тревог, откалиброванный на «чистых» данных (без вставленных шоков)."""
    th = {}
    for name, S in scores.items():
        v = S[t_from:].ravel()
        v = v[~np.isnan(v)]
        th[name] = {far: float(np.quantile(v, 1 - far)) for far in far_levels}
    return th


def evaluate(scores: dict[str, np.ndarray], meta: pd.DataFrame, th: dict, t_from: int, cfg: dict) -> pd.DataFrame:
    """Метрики детекторов на одном полусинтетическом прогоне."""
    b = cfg["budget"]
    win = cfg["synth"]["detect_window"]
    n_t, n = next(iter(scores.values())).shape
    shocked = np.zeros(n, bool)
    shocked[meta.idx.values] = True
    onset = np.full(n, 10 ** 6)
    onset[meta.idx.values] = meta.onset_idx.values
    tt = np.arange(n_t)[:, None]
    label = (tt >= onset[None, :]) & shocked[None, :]
    clean_cols = ~shocked
    rows = []
    for name, S in scores.items():
        Sv = np.nan_to_num(S, nan=-np.inf)
        base = {"detector": name}
        # ранжирование MO-месяцев (PR-AUC) без порога
        from sklearn.metrics import average_precision_score
        mask = np.zeros_like(label)
        mask[t_from:] = True
        y_true = label[mask]
        y_score = np.nan_to_num(S[mask], nan=-1e9)
        base["PR_AUC"] = float(average_precision_score(y_true, y_score))
        # точность очереди аналитика: топ-k МО в каждом месяце после первого начала шока
        first = int(meta.onset_idx.min())
        for k in b["k_list"]:
            prec = []
            for t in range(first, n_t):
                top = np.argsort(-Sv[t])[:k]
                prec.append(label[t, top].mean())
            base[f"precision@{k}"] = float(np.mean(prec))
        for far in b["far_levels"]:
            alarm = Sv >= th[name][far]
            alarm[:t_from] = False
            fa = alarm[t_from:, clean_cols]
            det, delays = [], []
            for r in meta.itertuples():
                a = alarm[r.onset_idx: r.onset_idx + win, r.idx]
                hit = np.flatnonzero(a)
                det.append(len(hit) > 0)
                if len(hit):
                    delays.append(hit[0])
            det = np.array(det)
            n_false = int(fa.sum())
            n_true_alarm = int((alarm & label).sum())
            n_alarm = int(alarm[t_from:].sum())
            f = int(far * 100)
            base[f"recall@{f}"] = float(det.mean())
            for size in sorted(meta["size"].unique()):
                base[f"recall@{f}_size{int(size * 100)}"] = float(det[meta["size"].values == size].mean())
            base[f"recall@{f}_cluster"] = float(det[meta.cluster.values].mean()) if meta.cluster.any() else np.nan
            base[f"recall@{f}_isolated"] = float(det[~meta.cluster.values].mean())
            base[f"real_FAR@{f}"] = float(fa.mean())
            base[f"precision@FAR{f}"] = float(n_true_alarm / max(n_alarm, 1))
            base[f"median_delay@{f}"] = float(np.median(delays)) if delays else np.nan
            base[f"cost@{f}_per1000MO"] = float((b["cost_miss"] * (~det).sum() + b["cost_false_alarm"] * n_false)
                                                / n * 1000)
        rows.append(base)
    return pd.DataFrame(rows)
