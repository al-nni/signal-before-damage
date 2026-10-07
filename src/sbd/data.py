"""Загрузка и проверка данных.

Схема consumption.parquet (проверена по файлу и по описанию в архиве):
    date (str 'YYYY-MM'), territory_id (int), category (str), value (int, руб. на жителя в месяц).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "configs/base.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    Path(cfg["paths"]["out"]).mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg["seed"])
    return cfg


def _ensure_unzipped(cfg: dict) -> Path:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    target = raw_dir / "consumption.parquet"
    if target.exists():
        return raw_dir
    z = Path(cfg["paths"]["raw_zip"])
    if not z.exists():
        raise FileNotFoundError(
            "Нет данных. Скачайте архив со страницы конкурса "
            "(https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip) "
            f"и положите его в {z}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if name.endswith(".parquet"):
                (raw_dir / Path(name).name).write_bytes(zf.read(name))
    return raw_dir


def load_long(cfg: dict) -> pd.DataFrame:
    raw_dir = _ensure_unzipped(cfg)
    df = pd.read_parquet(raw_dir / "consumption.parquet")
    expected = {"date", "territory_id", "category", "value"}
    if not expected.issubset(df.columns):
        raise KeyError(f"Ожидались колонки {expected}, в файле: {list(df.columns)}")
    df = df.rename(columns={"value": "y"})
    df["date"] = pd.PeriodIndex(df["date"].astype(str), freq="M")
    return df


def build_panel(cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Возвращает {категория: таблица месяцы x МО} и отчёт о данных.

    В панель попадают МО, у которых есть все 6 категорий за все 24 месяца.
    """
    df = load_long(cfg)
    dups = int(df.duplicated(["territory_id", "date", "category"]).sum())
    if dups:
        raise ValueError(f"В данных {dups} дублей (territory_id, date, category)")
    cats = cfg["data"]["categories"]
    n = df.groupby(["territory_id", "category"])["date"].nunique().unstack()
    full = n.index[(n.reindex(columns=cats) >= cfg["data"]["min_months"]).all(axis=1)]
    panel = {}
    for c in cats:
        w = (df[df.category == c].pivot(index="date", columns="territory_id", values="y")
             .sort_index()[sorted(full)].astype(float))
        if w.isna().any().any() or (w <= 0).any().any():
            raise ValueError(f"Пропуски или неположительные значения в категории {c}")
        panel[c] = w
    report = pd.Series({
        "строк в файле": len(df),
        "МО в файле": df.territory_id.nunique(),
        "МО с полной историей (6 категорий x 24 мес.)": len(full),
        "месяцев": df.date.nunique(),
        "период": f"{df.date.min()} – {df.date.max()}",
        "категорий": df.category.nunique(),
        "дублей": dups,
        "значений <= 0": int((df.y <= 0).sum()),
        "мин. значение, руб.": int(df.y.min()),
        "медиана, руб. (все категории)": float(df[df.category == cats[0]].y.median()),
    })
    return panel, report


def load_reference(cfg: dict, ids) -> pd.DataFrame:
    ref = pd.read_csv(cfg["paths"]["reference"]).set_index("territory_id")
    missing = set(ids) - set(ref.index)
    if missing:
        raise ValueError(f"Нет в справочнике: {sorted(missing)[:10]} ...")
    return ref.loc[list(ids)]


def load_national(cfg: dict) -> pd.DataFrame:
    """Ряды СберИндекса по России, месяцы x тип расходов (млрд руб.)."""
    n = pd.read_csv(cfg["paths"]["national"])
    n["date"] = pd.PeriodIndex(n["period"].str[:7], freq="M")
    return n.pivot(index="date", columns="type", values="value").sort_index()


def validate_reference(cfg: dict, ref: pd.DataFrame, sample: int = 200_000) -> dict:
    """Независимая проверка справочника: автодорожные расстояния СберИндекса (connection.parquet)
    должны быть согласованы с расстояниями по координатам центров МО из справочника."""
    raw_dir = _ensure_unzipped(cfg)
    conn = pd.read_parquet(raw_dir / "connection.parquet")
    h = conn[conn["type"] == "highway"]
    h = h.sample(min(sample, len(h)), random_state=cfg["seed"])
    h = h[h.territory_id_x.isin(ref.index) & h.territory_id_y.isin(ref.index)]

    def hav(a, b):
        la1, lo1, la2, lo2 = map(np.radians, [a[:, 0], a[:, 1], b[:, 0], b[:, 1]])
        x = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        return 2 * 6371 * np.arcsin(np.sqrt(x))

    ll = ref[["lat", "lon"]]
    d = hav(ll.loc[h.territory_id_x].values, ll.loc[h.territory_id_y].values)
    rng = np.random.default_rng(cfg["seed"])
    ll_s = ll.copy()
    ll_s.index = ll.index[rng.permutation(len(ll))]
    d_s = hav(ll_s.loc[h.territory_id_x].values, ll_s.loc[h.territory_id_y].values)
    ratio = h.distance.values / np.maximum(d, 1.0)
    return {
        "пар МО": int(len(h)),
        "корреляция дорога ~ прямая": float(np.corrcoef(h.distance.values, d)[0, 1]),
        "то же при перемешанных координатах": float(np.corrcoef(h.distance.values, d_s)[0, 1]),
        "медиана отношения дорога/прямая": float(np.median(ratio)),
        "доля пар, где дорога короче 0.9 прямой": float(np.mean(ratio < 0.9)),
    }


def validate_national(panel: dict, ref: pd.DataFrame, nat: pd.DataFrame, cfg: dict) -> dict:
    """Сверка ряда СберИндекса по России с суммой по панели (траты на жителя x население)."""
    tot = panel[cfg["data"]["total_category"]]
    pop = ref.population.reindex(tot.columns).fillna(ref.population.median())
    agg = (tot * pop.values).sum(axis=1)
    j = pd.concat([agg.rename("panel"), nat["Всего"].rename("nat")], axis=1).dropna()
    dl = np.log(j).diff().dropna()
    return {"месяцев сверки": int(len(j)),
            "корреляция месячных изменений (панель ~ Россия)": float(dl.corr().iloc[0, 1])}


def neighbours(cfg: dict, ids, km: float, k_max: int = 5) -> dict[int, list[int]]:
    """До k_max ближайших соседей МО по автодорогам в радиусе km (из connection.parquet)."""
    raw_dir = _ensure_unzipped(cfg)
    conn = pd.read_parquet(raw_dir / "connection.parquet")
    ids = set(int(i) for i in ids)
    h = conn[(conn["type"] == "highway") & (conn.distance <= km)
             & conn.territory_id_x.isin(ids) & conn.territory_id_y.isin(ids)]
    both = pd.concat([h[["territory_id_x", "territory_id_y", "distance"]],
                      h.rename(columns={"territory_id_x": "territory_id_y",
                                        "territory_id_y": "territory_id_x"})[
                          ["territory_id_x", "territory_id_y", "distance"]]])
    both = both.sort_values(["territory_id_x", "distance"]).groupby("territory_id_x").head(k_max)
    nb: dict[int, list[int]] = {i: [] for i in ids}
    for x, y in zip(both.territory_id_x.values, both.territory_id_y.values):
        nb[int(x)].append(int(y))
    return nb
