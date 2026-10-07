"""Презентация 16:9 (PDF), 11 слайдов. Числа — из outputs/ через results.py."""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import Frame, Paragraph

import results as RS
from pdfkit import ACCENT, ACCENT_T, INK, INK2, RULE, TINT, fmt, fsign

ROOT = Path(__file__).resolve().parents[1]
FIG = RS.OUT / "figures"
Wp, Hp = 960, 540
M = 44
DET_RU = {
    "z_total": "z-оценка месяца", "cusum_total": "CUSUM", "ewma_total": "EWMA", "bocpd_total": "BOCPD",
    "binseg_total": "BinSeg", "pelt_total": "PELT", "stouffer": "Стауффер по 6 категориям",
    "stouffer_cusum": "CUSUM по Стауфферу", "spatial_stouffer": "Стауффер по соседям",
    "stouffer_or_spatial": "Стауффер: МО или соседи",
}

BODY = ParagraphStyle("b", fontName="Inter-Regular", fontSize=15, leading=21, textColor=INK, spaceAfter=9)
SMALL = ParagraphStyle("s", fontName="Inter-Regular", fontSize=11.5, leading=16, textColor=INK2, spaceAfter=6)
BUL = ParagraphStyle("bu", parent=BODY, leftIndent=16, bulletIndent=0, spaceAfter=10)
CARD_T = ParagraphStyle("ct", fontName="Inter-SemiBold", fontSize=15, leading=19, textColor=INK, spaceAfter=6)
CARD_B = ParagraphStyle("cb", fontName="Inter-Regular", fontSize=14, leading=19.5, textColor=INK2)


class Deck:
    def __init__(self, path):
        self.c = rl_canvas.Canvas(str(path), pagesize=(Wp, Hp))
        self.c.setTitle("Сигнал до ущерба — презентация")
        self.c.setAuthor("Алина Антошкина, Полина Протасова")
        self.n = 0

    def new(self, title=None, kicker=None):
        if self.n:
            self._footer()
            self.c.showPage()
        self.n += 1
        c = self.c
        c.setFillColor(colors.white)
        c.rect(0, 0, Wp, Hp, stroke=0, fill=1)
        if title:
            c.setFillColor(ACCENT)
            c.rect(M, Hp - 52, 28, 4, stroke=0, fill=1)
            if kicker:
                c.setFont("Inter-Medium", 11.5)
                c.setFillColor(INK2)
                c.drawString(M + 36, Hp - 55, kicker.upper())
            from reportlab.pdfbase.pdfmetrics import stringWidth
            if stringWidth(title, "Inter-SemiBold", 27) > Wp - 2 * M:
                raise RuntimeError(f"Заголовок слайда {self.n} длиннее ширины: {title}")
            c.setFont("Inter-SemiBold", 27)
            c.setFillColor(INK)
            c.drawString(M, Hp - 92, title)

    def _footer(self):
        c = self.c
        c.setFont("Inter-Regular", 9)
        c.setFillColor(INK2)
        c.drawString(M, 20, "Сигнал до ущерба · онлайн-конкурс СберИндекса 2026 · «Прогнозирование»")
        c.drawRightString(Wp - M, 20, str(self.n))

    def paras(self, items, x, y_top, w, h, style=BODY, bullets=False):
        f = Frame(x, y_top - h, w, h, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0)
        fl = [Paragraph(t, BUL if bullets else style, bulletText="•" if bullets else None) for t in items]
        f.addFromList(fl, self.c)
        if fl:
            raise RuntimeError(f"Текст не поместился на слайде {self.n}: {fl[0].text[:60]}")

    def image(self, name, x, y_top, w=None, h=None):
        p = FIG / name
        iw, ih = PImage.open(p).size
        if w is not None and h is not None:
            s = min(w / iw, h / ih)
            w, h = iw * s, ih * s
        elif w is not None:
            h = w * ih / iw
        else:
            w = h * iw / ih
        self.c.drawImage(str(p), x, y_top - h, w, h, preserveAspectRatio=True, mask="auto")
        return w, h

    def big(self, x, y, value, label, w=200, color=INK):
        c = self.c
        c.setFont("Inter-SemiBold", 38)
        c.setFillColor(color)
        c.drawString(x, y, value)
        f = Frame(x, y - 66, w, 58, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        fl = [Paragraph(label, SMALL)]
        f.addFromList(fl, c)
        if fl:
            raise RuntimeError(f"Подпись не поместилась на слайде {self.n}: {label[:50]}")

    def card(self, x, y_top, w, h, title, body, bg=TINT):
        c = self.c
        c.setFillColor(bg)
        c.roundRect(x, y_top - h, w, h, 8, stroke=0, fill=1)
        f = Frame(x + 14, y_top - h + 10, w - 28, h - 24, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        fl = [Paragraph(title, CARD_T), Paragraph(body, CARD_B)]
        f.addFromList(fl, c)
        if fl:
            raise RuntimeError(f"Карточка не поместилась на слайде {self.n}: {title}")

    def box(self, x, y, w, h, text, bg=TINT, fg=INK, bold=False):
        c = self.c
        c.setFillColor(bg)
        c.roundRect(x, y, w, h, 7, stroke=0, fill=1)
        st = ParagraphStyle("bx", fontName="Inter-SemiBold" if bold else "Inter-Medium", fontSize=12.5, leading=15.5,
                            textColor=fg, alignment=1)
        p = Paragraph(text, st)
        pw, ph = p.wrap(w - 16, h)
        p.drawOn(c, x + 8, y + (h - ph) / 2)

    def arrow(self, x1, y1, x2, y2):
        c = self.c
        c.setStrokeColor(INK2)
        c.setFillColor(INK2)
        c.setLineWidth(1.4)
        c.line(x1, y1, x2, y2)
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        L = 7
        p = c.beginPath()
        p.moveTo(x2, y2)
        p.lineTo(x2 - L * math.cos(a - 0.4), y2 - L * math.sin(a - 0.4))
        p.lineTo(x2 - L * math.cos(a + 0.4), y2 - L * math.sin(a + 0.4))
        p.close()
        c.drawPath(p, stroke=0, fill=1)

    def save(self):
        self._footer()
        self.c.save()


def _long_h_slide(R):
    worse12 = [c.lower() for c, v in RS.gain_cat(R, R["main"], 12).items() if v < 0]
    worse6 = [c.lower() for c, v in RS.gain_cat(R, R["main"], 6).items() if v < 0]
    if not worse12 and not worse6:
        return "На 6 и 12 мес. ансамбль тоже лучше Prophet во всех категориях."
    parts = []
    if worse6:
        parts.append("на 6 мес. — " + ", ".join(worse6))
    if worse12:
        parts.append("на 12 мес. — " + ", ".join(worse12))
    return ("Где Prophet точнее: " + "; ".join(parts) + ". Из точки 12.2023 модель не видит местного тренда.")


def build_slides(R, path):
    g, det = RS.g, RS.det
    D = Deck(path)
    main = R["main_det"]
    MAIN = R["main"]
    gain = {h: -g(R, MAIN, h, "MAE_vs_base_%") for h in (1, 3, 6, 12)}
    guided = R["guided"].set_index(["FAR вне зоны ЧС", "FAR в зоне ЧС"])
    gd = guided.loc[(0.03, 0.2)]
    n_named = len(R["flood_mo"])
    extra_pct = 100 * gd["дополнительных тревог за год"] / (gd["всего тревог за год"] - gd["дополнительных тревог за год"])

    # 1. титул
    D.new()
    c = D.c
    c.setFillColor(ACCENT)
    c.rect(M, Hp - 150, 40, 5, stroke=0, fill=1)
    c.setFont("Inter-SemiBold", 52)
    c.setFillColor(INK)
    c.drawString(M, Hp - 215, "Сигнал до ущерба")
    D.paras(["Прогноз безналичного потребления 2 016 муниципалитетов и раннее обнаружение шоков "
             "методами транзакционного мониторинга"], M, Hp - 240, 780, 100,
            ParagraphStyle("st", fontName="Inter-Regular", fontSize=21, leading=28, textColor=INK2))
    D.paras(["Алина Антошкина, Полина Протасова<br/>Финансовый университет при Правительстве РФ",
             "Онлайн-конкурс СберИндекса 2026 · направление «Прогнозирование»"], M, 150, 700, 90, SMALL)

    # 2. зачем
    D.new("Шок в регионе ломает «норму» клиентов", "зачем это банку")
    D.card(M, Hp - 125, 280, 215, "Что происходит",
           "Паводок, режим ЧС, закрытие работодателя: люди снимают наличные, переводят деньги "
           "родственникам, получают компенсации и тратят их.")
    D.card(M + 296, Hp - 125, 280, 215, "Чем это грозит",
           "Антифрод-правила блокируют легальные операции. Кредитные модели ещё смотрят на "
           "докризисные данные, а просрочка придёт позже.")
    D.card(M + 592, Hp - 125, 280, 215, "Что мы предлагаем",
           "Точный прогноз «нормы» трат по каждому МО и детектор отклонений, настроенный как "
           "антифрод: доля найденных шоков при фиксированном бюджете тревог.", bg=ACCENT_T)
    D.paras(["Шоки бывают в обе стороны: провал (ущерб, эвакуация) и всплеск (компенсации, "
             "восстановление) — детекторы у нас двусторонние."], M, 170, 870, 50, BODY)

    # 3. данные
    d = R["data"]
    ref = d["справочник: проверка по автодорожным расстояниям"]
    nat = d["ряд СберИндекса по России: сверка с панелью"]
    D.new("Данные: пять наборов СберИндекса и календарь событий", "данные")
    D.big(M, Hp - 170, "2 016", "МО с полной историей: 6 категорий × 24 месяца (2023–2024)", 230)
    D.big(M + 290, Hp - 170, fmt(ref["корреляция дорога ~ прямая"], 2),
          "согласие справочника МО с автодорожными расстояниями (0,00 при перемешивании)", 250)
    D.big(M + 590, Hp - 170, str(len(R["events"])),
          "событий: решения Банка России и режимы ЧС с датой, когда о них стало известно", 260)
    D.paras([
        "Траты на жителя по МО и категориям — цель прогноза и детекции",
        "Связи между МО — соседи для пространственного детектора",
        "Доступность рынков и справочник МО — признаки и регион",
        f"Ряд СберИндекса по России с 2018 года — сезонность для первой точки прогноза (согласие с панелью "
        f"{fmt(nat['корреляция месячных изменений (панель ~ Россия)'], 2)})",
    ], M, Hp - 265, 860, 200, bullets=True)

    # 4. архитектура
    D.new("Как устроено решение", "архитектура")
    y = Hp - 260
    xs = [M, M + 178, M + 356, M + 534, M + 712]
    labels = ["Траты МО<br/>× 6 категорий", "Общий фактор<br/>+ регион", "Прогноз нормы<br/>(ансамбль)",
              "Остатки z<br/>(локальный сдвиг)", "Детекторы<br/>+ бюджет тревог"]
    for i, (x, t) in enumerate(zip(xs, labels)):
        D.box(x, y, 150, 62, t, bg=ACCENT_T if i in (2, 4) else TINT, bold=i in (2, 4))
        if i:
            D.arrow(xs[i - 1] + 152, y + 31, x - 3, y + 31)
    D.box(M + 534, y - 130, 328, 56, "Новости: ставка ЦБ, режимы ЧС<br/>(только известные к концу месяца)", bg=TINT)
    D.arrow(M + 790, y - 72, M + 790, y - 3)
    D.arrow(M + 560, y - 72, M + 300, y - 3)
    D.box(M + 712, y + 110, 150, 56, "Антифрод и<br/>кредитный риск", bg=INK, fg=colors.white, bold=True)
    D.arrow(M + 787, y + 64, M + 787, y + 107)
    D.paras(["Каждый блок — модуль src/sbd/; весь расчёт — make all; тесты проверяют отсутствие утечки."],
            M, 118, 870, 40, SMALL)

    # 5. прогноз
    gs = [gain[h] for h in (1, 3, 6, 12)]
    D.new(f"Прогноз: ошибка на {fmt(min(gs))}–{fmt(max(gs))}% ниже, чем у Prophet", "прогноз")
    D.image("fig_mae_by_h.png", M + 330, Hp - 110, w=540)
    D.big(M, Hp - 170, f"−{fmt(gain[1])}%", f"MAE к Prophet на 1 мес.: {fmt(g(R, MAIN, 1))} против "
          f"{fmt(g(R, 'prophet', 1))} руб. на жителя", 290, color=ACCENT)
    D.big(M, Hp - 265, f"−{fmt(gain[3])}%", f"на 3 мес. · −{fmt(gain[6])}% на 6 мес.", 290)
    D.big(M, Hp - 360, f"{fmt(100 * g(R, MAIN, 1, 'share_MO_better'))}%",
          "муниципалитетов, где ансамбль точнее Prophet (тест Уилкоксона p &lt; 0,001)", 290)
    D.paras([("Итоговый ансамбль = среднее трёх прогнозов: «общий фактор + регион», «свой прошлогодний сезон "
              "+ рост» и Chronos-Bolt. " if MAIN == "ensemble_fm" else
              "Ансамбль = среднее «общий фактор + регион» и «свой прошлогодний сезон + рост». ")
             + "Горизонт 12 мес. — одна точка прогноза."], M, 110, 870, 40, SMALL)

    # 6. по категориям и что не сработало
    gc = RS.gain_cat(R, MAIN, 1)
    hb = RS.all_better_h(R, MAIN)
    hb_txt = "1–6" if hb[:3] == [1, 3, 6] else ("1–3" if hb[:2] == [1, 3] else "1")
    D.new(f"На {hb_txt} мес. выигрыш во всех категориях. Что не сработало", "прогноз")
    D.image("fig_gain_by_category.png", M, Hp - 110, w=500)
    D.paras([
        f"Главный источник точности — сезонность, перенесённая с 2 016 МО: у одного МО всего 24 точки.",
        f"Бустинг-поправка (LightGBM) хуже простого фактора: {fmt(g(R, 'lgbm', 3))} против "
        f"{fmt(g(R, 'factor_region', 3))} на 3 мес. — мало месяцев для обучения.",
        _long_h_slide(R),
    ], M + 530, Hp - 120, 340, 330, bullets=True, style=BODY)

    # 6а. фундаментальная модель
    if R["chronos"]:
        D.new("Фундаментальная модель работает без общей сезонности", "chronos-bolt")
        D.big(M, Hp - 170, fmt(g(R, "chronos_bolt_raw", 1)),
              "MAE Chronos-Bolt на сыром ряду МО (h = 1) — хуже Prophet: по 12–23 точкам сезонность "
              "не выучить", 260)
        D.big(M + 300, Hp - 170, fmt(g(R, "chronos_bolt", 1)),
              f"та же модель на отклонениях от общего фактора — лучшая одиночная модель на 1 мес. "
              f"(фактор + регион: {fmt(g(R, 'factor_region', 1))})", 270, color=ACCENT)
        D.big(M + 610, Hp - 170, fmt(g(R, "ensemble_fm", 1)),
              f"ансамбль с Chronos-Bolt на 1 мес. (без неё — {fmt(g(R, 'ensemble', 1))}; Prophet — "
              f"{fmt(g(R, 'prophet', 1))})", 260)
        D.paras([
            "Chronos-Bolt small (Amazon, Apache-2.0), zero-shot на CPU; подлинность весов проверена по SHA-256.",
            f"На 3–12 мес. модель немного уступает факторной ({fmt(g(R, 'chronos_bolt', 3))} против "
            f"{fmt(g(R, 'factor_region', 3))} на 3 мес.): она дополняет перенос сезонности с панели, а не заменяет его.",
        ], M, Hp - 300, 870, 160, bullets=True)

    # 7. детекторы
    D.new("Детекторы как антифрод: полнота при бюджете тревог", "шоки")
    D.image("fig_detect_curve.png", M + 330, Hp - 110, w=540)
    D.big(M, Hp - 170, f"{fmt(100 * det(R, main, 'recall@3'))}%",
          f"шоков пойманы в первые 3 месяца при 3% ложных тревог ({DET_RU[main]})", 290, color=ACCENT)
    best_cs = max(R["det"].index, key=lambda n: det(R, n, "precision@20"))
    D.big(M, Hp - 265, fmt(det(R, best_cs, "precision@20"), 2),
          f"точность очереди из 20 МО у «{DET_RU[best_cs]}»", 290)
    dl = det(R, main, "median_delay@3")
    D.big(M, Hp - 360, f"{fmt(dl, 1 if dl % 1 else 0)} мес.",
          "медианная задержка первой тревоги" + (" — тревога в месяц начала шока" if dl == 0 else ""), 290)
    D.paras(["Проверка: шоки ±5/10/20% вставлены в реальные ряды 10% МО, 5 повторов; порог калибруется на "
             "данных без вставок."], M, 110, 870, 40, SMALL)

    # 8. сила шока и цена
    D.new("Сильные шоки ловятся, слабые — нет", "шоки · порог выбирает цена ошибки")
    D.image("fig_recall_by_size.png", M, Hp - 110, w=430)
    D.image("fig_cost.png", M + 450, Hp - 110, w=430)
    D.paras([f"±20%: ловим {fmt(100 * det(R, main, 'recall@3_size-20'))}% провалов и "
             f"{fmt(100 * det(R, main, 'recall@3_size20'))}% всплесков; −5%: только "
             f"{fmt(100 * det(R, main, 'recall@3_size-5'))}%. PELT, BinSeg и BOCPD на 3–12 точках уступают "
             "простым статистикам по остаткам хорошего прогноза."], M, 130, 870, 60, SMALL)

    # 9. новости
    D.new("Новости дают детектору «адрес»", "новостной слой")
    D.paras([
        "Событие относится к месяцу, только если о нём было известно до конца месяца.",
        "Ставка ЦБ — признак общего фактора; режим ЧС — снижает порог для МО региона.",
    ], M, Hp - 120, 420, 150, bullets=True)
    D.big(M, Hp - 300, f"{int(gd['из них без подсказки'])} → {int(gd['названных МО с тревогой (апр–июн)'])}",
          f"из {n_named} МО, названных в сообщениях о паводке, получают тревогу в апреле–июне", 380, color=ACCENT)
    D.big(M + 470, Hp - 300, f"+{fmt(extra_pct)}%",
          f"тревог за год по всей стране (+{int(gd['дополнительных тревог за год'])} МО-месяцев)", 380)
    D.paras(["Порог 3% ложных тревог вне зоны ЧС и 20% — в регионах с действующим режимом ЧС. Без "
             "подсказки паводок в месячных данных от шума не отличить: мы показываем это открыто."],
            M, 150, 870, 60, SMALL)

    # 10. паводок
    fr = R["flood_res"]
    grp = "Оренбургская обл.: МО из постановлений о ЧС"
    v = lambda cat, m: float(fr[(fr.группа == grp) & (fr.категория == cat) & (fr.месяц == m)]["отклонение_%"].iloc[0])  # noqa: E731
    from build import _orsk_may
    D.new("Паводок 2024: в среднем тишина, внутри — сдвиги", "реальное событие")
    D.image("fig_flood_lines.png", M + 330, Hp - 110, w=540)
    D.paras([
        f"{int(fr[fr.группа == grp]['МО'].iloc[0])} МО с постановлениями о ЧС: в среднем {fsign(v('Все категории', '2024-04'), 1)}% в апреле и "
        f"{fsign(v('Все категории', '2024-05'), 1)}% в мае к прогнозу.",
        f"Орск: {fsign(_orsk_may(R), 1)}% в мае — всплеск (гипотеза: компенсации и восстановление).",
        "Звериноголовский: −5…−8% три месяца подряд.",
    ], M, Hp - 120, 310, 330, bullets=True, style=BODY)

    # 11. ценность
    D.new("Что это даёт банку", "итог")
    D.card(M, Hp - 125, 280, 230, "Антифрод",
           "На 1–3 месяца расширяем допустимый профиль операций в МО с тревогой или режимом ЧС — меньше "
           "ложных блокировок. 3% бюджета ≈ 60 МО-месяцев в месяц.", bg=ACCENT_T)
    D.card(M + 296, Hp - 125, 280, 230, "Кредитный риск",
           "Временная надбавка к риску и мониторинг портфеля в МО с подтверждённым провалом; сигнал "
           "приходит раньше просрочки.")
    D.card(M + 592, Hp - 125, 280, 230, "Воспроизводимость",
           "make all → все таблицы, графики и этот PDF. Тесты на утечку, фиксированные версии, seed 42. "
           "Chronos-Bolt подключается одной командой.")
    D.paras(["Ограничения: 24 месяца данных, горизонт 12 мес. — одна точка; безналичные траты клиентов "
             "Сбера; профиль шока во вставках — допущение. Подробно — в методологическом отчёте."],
            M, 150, 870, 60, SMALL)
    D.save()
