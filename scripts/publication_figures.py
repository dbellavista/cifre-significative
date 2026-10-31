"""Figures for the Italian series of posts, labels in Italian, same palette as the notebooks.

Figures are drawn for a phone screen (narrow, large text, few labels); see apply_mobile_style.

Usage: uv run python scripts/publication_figures.py [00 ...]   -> figures/NN_name.png; the numbers quoted in
the post are printed so they can be re-checked against the notebooks before publishing.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cifresignificative.analysis import industry_panel
from cifresignificative.analysis.data import (
    EU27_ISO3,
    ITALY,
    MARKET_SECTORS,
    PROCESSED,
    aggregate_productivity,
    gdp_pc_growth_decomposition,
    load_long,
    load_panel,
    load_sectors,
    load_sectors64,
    long_wide,
    shift_share,
)
from cifresignificative.analysis.synth import placebo_ratios, rmspe, synthetic_control
from cifresignificative.analysis.timeline import government_stats, load_governments
from cifresignificative.viz import style
from cifresignificative.viz.style import (
    EU_C,
    INK,
    INK2,
    ITALY_C,
    PEER_C,
    SURFACE,
    apply_style,
)

OUT = Path("figures")
def apply_mobile_style() -> None:
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 11.5, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11, "axes.titlelocation": "left"})
W_PX = 1080  # phone layout: figures are 5.4 in wide at 200 dpi, shown at about 360 px on a phone, so text is at least 11 pt
END = 2025  # last year of the post: 2025 is a first estimate, said in the text
NAMES = {"DEU": "Germania", "FRA": "Francia", "ESP": "Spagna", "GBR": "Regno Unito", "PRT": "Portogallo", "GRC": "Grecia", "AUT": "Austria", "BEL": "Belgio", "DNK": "Danimarca", "FIN": "Finlandia", "IRL": "Irlanda", "LUX": "Lussemburgo", "NLD": "Paesi Bassi", "SWE": "Svezia", "USA": "Stati Uniti", "JPN": "Giappone"}


def save(fig, name: str, note: str = "", dy: float = -0.02) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if note:
        fig.text(0.0, dy, note, fontsize=7.5, color=INK2, ha="left", va="top")
    fig.savefig(OUT / f"{name}.png", dpi=round(W_PX / 5.4), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def spread(vals: dict[str, float], gap: float) -> dict[str, float]:
    """Label heights at least `gap` apart (in data units), in the order of the values, so end-of-line labels do not overlap."""
    out, last = {}, -1e9
    for c, v in sorted(vals.items(), key=lambda kv: kv[1]):
        out[c] = max(v, last + gap); last = out[c]
    return out


def fig00_decomposition(panel) -> None:
    dec = gdp_pc_growth_decomposition(panel, 2005, END)
    parts = ["employment rate", "hours per worker", "productivity & demography (residual)"]
    labels = ["Occupazione", "Ore per lavoratore", "Residuo (produttività e demografia)"]
    show = dec.loc[["ITA", "DEU", "FRA", "ESP"]].copy()
    show.loc["Mediana\ndegli altri 26"] = dec.drop("ITA").median()
    show.index = [NAMES.get(i, "Italia" if i == "ITA" else i) for i in show.index]
    print(f"00 decomposition 2005-{END} ({len(dec)} countries, log points x100):\n", show.round(1).to_string())
    print("   rank of Italy by total (1 = lowest):", int(dec["real GDP per head"].rank()["ITA"]), "| lowest:", dec["real GDP per head"].nsmallest(4).round(1).to_dict())
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    cols = ["#2a78d6", "#eda100", "#8a8a85"]
    pos, neg = np.zeros(len(show)), np.zeros(len(show))
    for part, lab, col in zip(parts, labels, cols, strict=True):
        v = show[part].values; base = np.where(v >= 0, pos, neg)
        ax.barh(range(len(show)), v, left=base, color=col, height=0.62, label=lab, edgecolor=SURFACE, linewidth=2)
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.scatter(show["real GDP per head"], range(len(show)), color=INK, zorder=5, s=28, label="Totale (PIL reale per abitante)")
    for i, t in enumerate(show["real GDP per head"]): ax.annotate(f"{t:+.0f}", (t, i), xytext=(7, -11), textcoords="offset points", fontsize=12, color=INK, fontweight="bold")
    ax.set_yticks(range(len(show)), show.index); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("punti logaritmici (x100)")
    ax.set_title(f"Da dove viene la crescita\ndel PIL per abitante, 2005-{END}", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.16), ncol=1)
    save(fig, "00_decomposizione_crescita")


def fig00_long(lh) -> None:
    gdp = long_wide(lh, "gdp_real_per_head_ncu"); idx = gdp / gdp.loc[1995] * 100
    g = gdp.apply(np.log).diff() * 100
    dec = pd.DataFrame({f"{d}s": g.loc[d:d + 9].mean() for d in range(1960, 2020, 10)}); dec[f"2020-{END % 100}"] = g.loc[2020:END].mean()
    print("00 growth by decade (% a year):\n", dec.loc[[ITALY, "DEU", "FRA", "ESP"]].round(1).to_string())
    print(f"00 index 1995=100 in {END}:", idx.loc[END, [ITALY, "DEU", "FRA", "ESP", "GBR"]].round(1).to_dict())
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    peers = ["DEU", "FRA", "ESP"]; ypos = spread(idx.loc[END, peers].to_dict(), 9)  # end labels at least 4 index points apart
    for c in peers:
        ax.plot(idx.index, idx[c], color=PEER_C, lw=2)
        ax.annotate(f"{NAMES[c]} {idx.loc[END, c]:.0f}", (END, idx.loc[END, c]), xytext=(END + 2, ypos[c]), textcoords="data", fontsize=11, color=INK2, va="center", arrowprops={"arrowstyle": "-", "color": PEER_C, "lw": 0.6, "shrinkA": 0, "shrinkB": 0})
    ax.plot(idx.index, idx[ITALY], color=ITALY_C, lw=2.8)
    ax.annotate(f"Italia {idx.loc[END, ITALY]:.0f}", (END, idx.loc[END, ITALY]), xytext=(END + 2, idx.loc[END, ITALY] - 4), textcoords="data", color=ITALY_C, fontweight="bold", va="center", fontsize=12)
    ax.axvline(1999, color=INK2, lw=0.8, ls=":"); ax.annotate("euro", (1999.5, 30), fontsize=11, color=INK2)
    ax.axhline(100, color=INK2, lw=0.6, ls=":"); ax.annotate("1995 = 100", (1961, 104), fontsize=11, color=INK2)
    ax.set_title("Crescita del PIL reale per abitante\n(indice, 1995 = 100)", loc="left"); ax.set_ylabel("indice di crescita, non un livello"); ax.set_xlim(1960, END + 22); ax.set_xticks(range(1960, END, 20))
    save(fig, "00_pil_pro_capite_lungo_periodo")


def fig00_synth(lh) -> None:
    gdp = long_wide(lh, "gdp_real_per_head_ncu"); y = (np.log(gdp) - np.log(gdp.loc[1995])) * 100
    units = [c for c in y.columns if c != "EU27" and y.loc[1970:END, c].notna().all()]  # every country of the panel with a complete series from 1970
    donors = [c for c in units if c != ITALY]; pre = (1970, 1994); yy = y.loc[1970:END, units]
    w, synth = synthetic_control(yy, ITALY, pre, donors)
    gap = yy[ITALY] - synth
    ratios = placebo_ratios(yy, units, pre, (1995, END))
    print(f"00 synthetic ({len(donors)} donors): weights", {k: round(v, 2) for k, v in w[w > 0.01].sort_values(ascending=False).items()},
          f"| RMSPE pre {rmspe(yy[ITALY], synth, pre):.1f} | gap 2005 {gap.loc[2005]:.1f}, {END} {gap.loc[END]:.1f} | ratio Italy {ratios[ITALY]:.1f} rank {int(ratios.rank(ascending=False)[ITALY])} of {len(ratios)}")
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.plot(yy.index, yy[ITALY], color=ITALY_C, lw=2.8); ax.plot(synth.index, synth, color=EU_C, lw=2.2, ls="--")
    ax.annotate("Italia", (2014, yy.loc[2014, ITALY]), xytext=(2014, yy.loc[2014, ITALY] - 14), textcoords="data", color=ITALY_C, fontweight="bold", ha="center", va="top", fontsize=13)
    ax.annotate("Italia sintetica", (END, synth.loc[END]), xytext=(-4, 10), textcoords="offset points", color=EU_C, fontweight="bold", ha="right", va="bottom", fontsize=13)
    ax.axvline(1995, color=INK2, lw=0.8, ls=":"); ax.annotate("1995", (1995.8, -58), fontsize=11, color=INK2); ax.set_xlim(1970, END + 1)
    ax.set_title("PIL reale per abitante:\nItalia e Italia sintetica", loc="left"); ax.set_ylabel("punti logaritmici (1995 = 0)")
    save(fig, "00_italia_sintetica")
    gaps: dict[str, pd.Series] = {}; pre_err: dict[str, float] = {}; post_err: dict[str, float] = {}  # every unit against its own mix of the others
    for u in units:
        _, s = synthetic_control(yy, u, pre, [c for c in units if c != u]); gaps[u] = yy[u] - s; pre_err[u] = rmspe(yy[u], s, pre); post_err[u] = rmspe(yy[u], s, (1995, END))
    cut = float(np.median(list(pre_err.values())))
    print(f"00 placebo split at the median pre-1995 error {cut:.2f}:")
    for u in sorted(units, key=lambda u: pre_err[u]): print(f"   {u} pre {pre_err[u]:.1f} post {post_err[u]:.1f} ratio {ratios[u]:.1f} gap {END} {gaps[u].loc[END]:.1f}")
    for name, title, members in [("00_placebo_ben_riprodotti", "Paesi riprodotti bene prima del 1995", [u for u in units if pre_err[u] <= cut]), ("00_placebo_mal_riprodotti", "Paesi riprodotti male prima del 1995", [u for u in units if pre_err[u] > cut])]:
        fig, ax = plt.subplots(figsize=(5.4, 5.6)); others = [u for u in members if u != ITALY]
        ends = spread({u: float(gaps[u].loc[END]) for u in others}, 5.5)
        for u in others:
            gp = gaps[u]; ax.plot(gp.index, gp, color=PEER_C, lw=1.8)
            ax.annotate(NAMES.get(u, u), (END, gp.loc[END]), xytext=(END + 2, ends[u]), textcoords="data", fontsize=11, color=INK2, va="center", arrowprops={"arrowstyle": "-", "color": PEER_C, "lw": 0.5, "shrinkA": 0, "shrinkB": 0})
        ax.plot(gap.index, gap, color=ITALY_C, lw=3); ax.annotate("Italia", (END, gap.loc[END]), xytext=(END + 2, gap.loc[END]), color=ITALY_C, fontweight="bold", va="center", fontsize=12)
        ax.axhline(0, color=INK2, lw=0.8); ax.axvline(1995, color=INK2, lw=0.8, ls=":"); ax.set_xlim(1970, END + 20); ax.set_xticks(range(1970, END, 10))
        ax.set_title(title.replace(" prima", "\nprima"), loc="left"); ax.set_ylabel("scarto dalla propria versione\nsintetica (punti logaritmici)")
        save(fig, name)


def fig00_governments(lh) -> None:
    stats = government_stats(lh, load_governments("1989-01-01"), "1989-01-01")
    st = stats.dropna(subset=["gdp_growth_pa_italy", "gdp_growth_pa_others"]).reset_index(drop=True)
    slower = int((st.gdp_growth_pa_italy < st.gdp_growth_pa_others).sum())
    print(f"00 governments: slower under {slower} of {len(st)}; faster under:", st[st.gdp_growth_pa_italy >= st.gdp_growth_pa_others].government.tolist())
    print(st[["government", "start", "months", "gdp_growth_pa_italy", "gdp_growth_pa_others"]].round(1).to_string())
    diff = st.gdp_growth_pa_italy - st.gdp_growth_pa_others; yy = np.arange(len(st))  # one bar per government: Italy minus the median of the others
    fig, ax = plt.subplots(figsize=(5.4, 0.3 * len(st) + 1.8))
    ax.barh(yy, diff, height=0.65, color=[ITALY_C if d >= 0 else "#e34948" for d in diff])
    ax.set_yticks(yy, [f"{g} ({s.year})" for g, s in zip(st.government, st.start, strict=True)], fontsize=11); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(f"Crescita dell'Italia meno mediana UE,\npunti % l'anno, per governo (Italia più lenta: {slower} su {len(st)})", loc="left", fontsize=12); ax.set_ylim(len(st) - 0.4, -0.7)
    ax.set_xlabel("punti percentuali l'anno (rosso: Italia più lenta)")
    save(fig, "00_crescita_per_governo")


def cover00(panel) -> None:
    """Share/preview image 1200x630: the two numbers of the post, readable at thumbnail size and under a centre crop."""
    tot = gdp_pc_growth_decomposition(panel, 2005, END)["real GDP per head"]; it, eu = np.expm1(tot[ITALY] / 100) * 100, np.expm1(tot.drop(ITALY).median() / 100) * 100
    print(f"00 cover: Italy {it:+.1f}%, median of the other {len(tot) - 1} {eu:+.1f}% (real GDP per head 2005-{END})")
    bg, fg, dim = "#14181f", "#ffffff", "#9aa3b2"
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=bg); ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 12); ax.set_ylim(0, 6.3); ax.axis("off")
    ax.text(6, 5.55, "L'Italia non è povera, è ferma", color=fg, fontsize=40, fontweight="bold", ha="center", va="center")
    ax.text(6, 4.85, f"PIL reale per abitante, 2005 - {END}", color=dim, fontsize=24, ha="center", va="center")
    for y, val, col, who in [(3.45, it, ITALY_C, "Italia"), (1.45, eu, EU_C, "Mediana UE")]:  # series palette, same as the figures
        ax.text(0.8, y, f"{val:+.0f}%".replace("-", "−"), color=col, fontsize=88, fontweight="bold", ha="left", va="center")
        ax.text(5.4, y + 0.42, who, color=col, fontsize=44, fontweight="bold", ha="left", va="center")
        ax.barh(y - 0.38, 5.6 * val / eu, left=5.4, height=0.5, color=col)
    ax.text(11.0, 0.35, "Cifre significative", color=dim, fontsize=15, ha="right", va="center")
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / "00_copertina.png", dpi=100, facecolor=bg); plt.close(fig)


SECTORS_IT = {"A": "Agricoltura", "B-E": "Industria", "F": "Costruzioni", "G-I": "Commercio, trasporti,\nalloggio e ristorazione", "J": "Informazione e\ncomunicazione", "K": "Finanza e assicurazioni",
              "M_N": "Servizi professionali\ne amministrativi", "O-Q": "PA, istruzione, sanità", "R-U": "Arte e altri servizi"}
INDUSTRIES_IT = {"F": "Costruzioni", "C26": "Computer ed elettronica", "A01": "Agricoltura e allevamento", "M69_M70": "Servizi legali e contabili", "G47": "Commercio al dettaglio", "D": "Elettricità e gas",
                 "I": "Alloggio e ristorazione", "H49": "Trasporto terrestre", "E37-E39": "Rifiuti e fognature", "M71": "Architettura e ingegneria", "G46": "Commercio all'ingrosso", "C13-C15": "Tessile e abbigliamento", "K64": "Servizi finanziari",
                 "H52": "Logistica e magazzinaggio", "J61": "Telecomunicazioni", "N79": "Agenzie di viaggio", "C28": "Macchinari", "M74_M75": "Altri servizi professionali", "C19": "Raffinazione del petrolio", "C24": "Metallurgia di base", "H50": "Trasporto marittimo", "H53": "Poste e corrieri", "G45": "Commercio e riparazione auto", "R93": "Sport e ricreazione", "C33": "Riparazione di macchinari"}
SECT_A, SECT_B = 2005, 2025  # nine sections: 25 members have both years (2025 is a first estimate, Bulgaria and Latvia have no 2025 value added)
IND_B = 2024  # A*64 industries: 2025 exists for few industries yet (Italy 15 of 57)


def _sector_sample(s: pd.DataFrame) -> list[str]:
    """EU27 members with a complete shift-share for the two years (all nine sections)."""
    return [c for c in EU27_ISO3 if c in s.iso3.unique() and shift_share(s, c, SECT_A, SECT_B).notna().all()]


def fig01_index(s: pd.DataFrame, others: list[str]) -> None:
    agg = aggregate_productivity(s)
    cov = agg[[c for c in agg if c in EU27_ISO3]].notna().sum(axis=1); last = SECT_B
    idx = agg.loc[:last] / agg.loc[SECT_A] * 100; med = idx[others].median(axis=1)
    allmed = (agg.loc[:last] / agg.loc[SECT_A] * 100)[[c for c in EU27_ISO3 if c != ITALY and c in agg]].median(axis=1).loc[last]
    print(f"01 (all members with a series in {last}: median {allmed:.1f}; the sample here: {len(others)} others)")
    print(f"01 index 2005=100 in {last}: Italy {idx.loc[last, ITALY]:.1f}, median of {len(others)} others {med.loc[last]:.1f}, DEU {idx.loc[last, 'DEU']:.1f}, FRA {idx.loc[last, 'FRA']:.1f}, ESP {idx.loc[last, 'ESP']:.1f}; coverage {cov.loc[last]} members")
    idx, med = idx.loc[SECT_A:], med.loc[SECT_A:]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for c in ["FRA", "ESP"]: ax.plot(idx.index, idx[c], color=PEER_C, lw=1.6)
    ax.plot(idx.index, idx["DEU"], color=PEER_C, lw=2.2); ax.plot(med.index, med, color=EU_C, lw=2.6, ls="--"); ax.plot(idx.index, idx[ITALY], color=ITALY_C, lw=2.8)
    ends = spread({"MED": med.loc[last], "DEU": idx.loc[last, "DEU"], ITALY: idx.loc[last, ITALY]}, 5.5)
    for k, lab, col, v in [("MED", "Mediana UE", EU_C, med.loc[last]), ("DEU", "Germania", INK2, idx.loc[last, "DEU"]), (ITALY, "Italia", ITALY_C, idx.loc[last, ITALY])]:
        ax.annotate(f"{lab} {v:.0f}", (last, v), xytext=(last + 0.7, ends[k]), textcoords="data", color=col, fontweight="bold" if k != "DEU" else "normal", fontsize=11.5, va="center")
    ax.axhline(100, color=INK2, lw=0.6, ls=":"); ax.set_xlim(SECT_A, last + 6.5); ax.set_xticks(range(2005, last, 5))
    ax.set_title("Valore aggiunto reale per ora lavorata\n(indice, 2005 = 100)", loc="left"); ax.set_ylabel("indice di crescita")
    save(fig, "01_produttivita_oraria")


def fig01_shift(s: pd.DataFrame, others: list[str]) -> None:
    dec = pd.DataFrame({c: shift_share(s, c, SECT_A, SECT_B) for c in [ITALY, "DEU", "FRA", "ESP", *others]}).T
    parts = ["within-sector productivity", "reallocation of hours", "interaction"]; labels = ["Dentro i settori", "Spostamento di ore tra settori", "Interazione"]
    show = dec.loc[[ITALY, "DEU", "FRA", "ESP"]].copy(); show.loc[f"Mediana\ndegli altri {len(others)}"] = dec.drop(ITALY).median()
    tot = show.sum(axis=1); tot.iloc[-1] = dec.drop(ITALY).sum(axis=1).median()  # median of totals: medians of the parts do not add up
    show.index = [NAMES.get(i, "Italia" if i == ITALY else i) for i in show.index]
    print(f"01 shift-share {SECT_A}-{SECT_B} (% of 2005 output per hour, {len(others)} others plus Italy):\n", show.round(1).assign(total=tot.values.round(1)).to_string())
    fig, ax = plt.subplots(figsize=(5.4, 5.2)); cols = ["#2a78d6", "#eda100", "#8a8a85"]; pos, neg = np.zeros(len(show)), np.zeros(len(show))
    for part, lab, col in zip(parts, labels, cols, strict=True):
        v = show[part].values; base = np.where(v >= 0, pos, neg)
        ax.barh(range(len(show)), v, left=base, color=col, height=0.62, label=lab, edgecolor=SURFACE, linewidth=2); pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.scatter(tot.values, range(len(show)), color=INK, zorder=5, s=28, label="Totale")
    for i, (t, end) in enumerate(zip(tot.values, pos, strict=True)): ax.annotate(f"totale {t:+.1f}%", (max(end, t), i), xytext=(6, 0), textcoords="offset points", fontsize=11.5, color=INK, fontweight="bold", va="center")
    ax.set_xlim(None, pos.max() + 9); ax.set_yticks(range(len(show)), show.index); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8); ax.set_xlabel("variazione % della produttività oraria aggregata")
    ax.set_title(f"Da dove viene la crescita della\nproduttività oraria, {SECT_A}-{SECT_B}", loc="left"); ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.16), ncol=1)
    save(fig, "01_scomposizione_settori")


def fig01_sectors(s: pd.DataFrame, others: list[str]) -> None:
    p = s.pivot_table(index="iso3", columns=["nace_r2", "year"], values="prod")
    g = pd.DataFrame({n: np.log(p[(n, SECT_B)] / p[(n, SECT_A)]) / (SECT_B - SECT_A) * 100 for n in MARKET_SECTORS}); gi = g.loc[ITALY]; gm = g.loc[others].median()
    cnt = g.loc[others].notna().sum()
    print(f"01 sector growth % a year {SECT_A}-{SECT_B} (log): Italy / median of others / n\n", pd.DataFrame({"ITA": gi, "med": gm, "n": cnt}).round(2).to_string(), "\n   Italy below median in", int((gi < gm).sum()), "of", len(gi))
    order = gi.sort_values().index; fig, ax = plt.subplots(figsize=(5.4, 6.4)); yy = np.arange(len(order))
    ax.barh(yy, gi[order], color=ITALY_C, height=0.5, zorder=2); ax.scatter(gm[order], yy, color=EU_C, marker="D", s=46, zorder=3)
    xl = max(gi.max(), gm.max()) + 1.25; xr = xl + 1.15; ax.set_xlim(min(gi.min(), gm.min()) - 0.3, xr + 0.65)
    for y, n in zip(yy, order, strict=True):
        ax.annotate(f"{gi[n]:+.1f}".replace("-0.0", "0.0"), (xl, y), color=ITALY_C, fontweight="bold", fontsize=11.5, ha="right", va="center"); ax.annotate(f"{gm[n]:+.1f}".replace("-0.0", "0.0"), (xr, y), color=EU_C, fontweight="bold", fontsize=11.5, ha="right", va="center")
    ax.annotate("Italia", (xl, -0.9), color=ITALY_C, fontweight="bold", fontsize=11.5, ha="right", va="center"); ax.annotate("Mediana", (xr + 0.1, -0.9), color=EU_C, fontweight="bold", fontsize=11.5, ha="right", va="center")
    ax.set_ylim(len(order) - 0.5, -1.4); ax.set_yticks(yy, [SECTORS_IT[n] for n in order]); ax.axvline(0, color=INK2, lw=0.8); ax.set_xticks(range(-1, 4))
    ax.set_title(f"Produttività oraria per settore,\ncrescita media annua {SECT_A}-{SECT_B}", loc="left"); ax.set_xlabel("% l'anno (barra: Italia, rombo: mediana UE)")
    save(fig, "01_crescita_per_settore")


def fig01_industries() -> None:
    """Within-industry gap against the median of the others, 62 market industries (the accounting of notebook 13), ten largest shortfalls."""
    s64 = load_sectors64(); pv = s64.pivot_table(index="iso3", columns=["nace_r2", "year"], values="prod"); hv = s64.pivot_table(index="iso3", columns=["nace_r2", "year"], values="hours_ths")
    inds = sorted({n for n, _ in pv.columns})

    def within(iso: str) -> pd.Series:
        rows = []
        for n in inds:
            try: p0, p1, h0, h1 = pv.loc[iso, (n, SECT_A)], pv.loc[iso, (n, IND_B)], hv.loc[iso, (n, SECT_A)], hv.loc[iso, (n, IND_B)]
            except KeyError: continue
            if min(p0, p1, h0, h1) > 0 and np.isfinite([p0, p1, h0, h1]).all(): rows.append((n, p0, p1, h0, h1))
        d = pd.DataFrame(rows, columns=["ind", "p0", "p1", "h0", "h1"]).set_index("ind"); s0 = d.h0 / d.h0.sum()
        return s0 * (d.p1 - d.p0) / (s0 * d.p0).sum() * 100
    eu = [c for c in EU27_ISO3 if c in pv.index and c != ITALY]; wi = within(ITALY); wm = pd.DataFrame({c: within(c) for c in eu}).median(axis=1)
    gap = (wi - wm.reindex(wi.index)).dropna().sort_values(); worst = gap.head(10)
    print(f"01 industries: within-industry Italy {wi.sum():.1f}, sum of medians {wm.sum():.1f}, total gap {gap.sum():.1f}, ten largest {worst.sum():.1f} ({worst.sum() / gap.sum() * 100:.0f}%), {len(eu)} others\n", gap.round(2).head(10).to_dict(), "\n   best:", gap.tail(4).round(2).to_dict())
    fig, ax = plt.subplots(figsize=(5.4, 4.8)); yy = np.arange(len(worst))
    ax.barh(yy, worst.values, color="#e34948", height=0.62)
    for y, v in zip(yy, worst.values, strict=True): ax.annotate(f"{v:.1f}", (0, y), xytext=(-5, 0), textcoords="offset points", ha="right", va="center", fontsize=11, color="white", fontweight="bold")
    ax.set_yticks(yy, [INDUSTRIES_IT.get(n, n) for n in worst.index]); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(f"Le dieci industrie con il maggiore scarto\n(su {len(gap)}), {SECT_A}-{IND_B}", loc="left"); ax.set_xlabel("punti % della produttività oraria aggregata,\nItalia meno mediana degli altri")
    save(fig, "01_scarto_per_industria")
    best = gap.tail(10).iloc[::-1]; print(f"   ten best: {best.round(2).to_dict()} | above the median: {int((gap > 0).sum())} of {len(gap)}; sum of the ten {best.sum():.1f}; sum of all positives {gap[gap > 0].sum():.2f}, of all negatives {gap[gap < 0].sum():.2f}")
    fig, ax = plt.subplots(figsize=(5.4, 4.8)); yy = np.arange(len(best)); ax.barh(yy, best.values, color=ITALY_C, height=0.62)
    for y, v in zip(yy, best.values, strict=True): ax.annotate(f"+{v:.1f}", (max(v, 0), y), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_yticks(yy, [INDUSTRIES_IT.get(n, n) for n in best.index]); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8); ax.set_xlim(-0.1, 2.0)
    ax.set_title(f"Le dieci industrie con il maggiore vantaggio\n(su {len(gap)}), {SECT_A}-{IND_B}", loc="left"); ax.set_xlabel("punti % della produttività oraria aggregata,\nItalia meno mediana degli altri")
    save(fig, "01_vantaggio_per_industria")


SEC_IT = {"A": "Agricoltura", "B": "Estrattive", "C": "Manifattura", "D-E": "Energia, acqua, rifiuti", "F": "Costruzioni", "G": "Commercio", "H": "Trasporti", "I": "Alloggio e rist.", "J": "Informazione e com.", "K": "Finanza", "M-N": "Servizi professionali", "R-S": "Arte e altri servizi"}


def fig01_myths() -> None:
    """Three common beliefs against the data: growth of output per hour 2005-2024, Italy and the median of the others, for a few industries."""
    s64 = load_sectors64(); p = s64.pivot_table(index="iso3", columns=["nace_r2", "year"], values="prod"); eu = [c for c in EU27_ISO3 if c in p.index and c != ITALY]
    g = lambda n: np.log(p[(n, IND_B)] / p[(n, SECT_A)]) / (IND_B - SECT_A) * 100
    rows = [("Alta tecnologia", None), ("Computer ed elettronica", "C26"), ("Servizi informatici", "J62_J63"), ("Telecomunicazioni", "J61"), ("Turismo", None), ("Alloggio e ristorazione", "I"), ("Dove l'Italia va bene", None), ("Farmaceutica", "C21")]
    vals = {n: (g(n)[ITALY], g(n)[eu].median(), int(g(n)[eu].notna().sum())) for _, n in rows if n}
    print(f"01 myths {SECT_A}-{IND_B}, % a year (log), Italy / median / n others:", {k: tuple(round(x, 2) for x in v) for k, v in vals.items()})
    fig, ax = plt.subplots(figsize=(5.4, 4.8)); lo, hi = min(min(v[:2]) for v in vals.values()), max(max(v[:2]) for v in vals.values()); xl = hi + 1.5; xr = xl + 1.75
    ax.set_xlim(lo - 0.4, xr + 0.2); ax.set_ylim(len(rows) - 0.4, -1.3); yy = {}
    for i, (lab, n) in enumerate(rows):
        if n is None: ax.annotate(lab, (lo - 0.4, i), fontsize=12, fontweight="bold", color=INK, va="center"); continue
        it, md, _ = vals[n]; yy[n] = i
        ax.barh(i, it, color=ITALY_C, height=0.5, zorder=2); ax.scatter([md], [i], color=EU_C, marker="D", s=46, zorder=3)
        ax.annotate(f"{it:+.1f}", (xl, i), color=ITALY_C, fontweight="bold", fontsize=11.5, ha="right", va="center"); ax.annotate(f"{md:+.1f}", (xr, i), color=EU_C, fontweight="bold", fontsize=11.5, ha="right", va="center")
    ax.annotate("Italia", (xl, -0.9), color=ITALY_C, fontweight="bold", fontsize=11.5, ha="right", va="center"); ax.annotate("Mediana", (xr + 0.1, -0.9), color=EU_C, fontweight="bold", fontsize=11.5, ha="right", va="center")
    ax.set_yticks([yy[n] for _, n in rows if n], [lab for lab, n in rows if n]); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(f"Tre luoghi comuni\nProduttività oraria, crescita media annua {SECT_A}-{IND_B}", loc="left", fontsize=12); ax.set_xlabel("% l'anno (barra: Italia, rombo: mediana UE)"); ax.set_xticks(range(-2, 5))
    save(fig, "01_luoghi_comuni")


def fig01_ict() -> None:
    """Italy's growth gap per KLEMS section against the section's ICT-capital intensity elsewhere (benchmark of notebook 28), with an illustrative line."""
    ek = pd.read_parquet(PROCESSED / "euklems_growth.parquet"); s = load_sectors64()
    G = industry_panel.industry_growth(s, *industry_panel.WINDOWS["full"]); G = G[G.iso3.isin(EU27_ISO3)]
    it = G[G.iso3 == ITALY].set_index("nace_r2").g; md = G[G.iso3 != ITALY].groupby("nace_r2").g.median()
    d = pd.DataFrame({"gap": (it - md) * 100}).dropna(); d["sec"] = d.index.map(industry_panel.section_of)
    x = industry_panel.intensity(ek, "LP1ConTangICT") * 100  # hundredths of a point a year of labour-productivity growth contributed by ICT capital, median of the other members, 1996-2004
    t = d.groupby("sec").agg(gap=("gap", "mean"), n=("gap", "size")).join(x.rename("ict")).dropna()
    print(f"01 ICT figure: {len(d)} industries in {len(t)} sections; Pearson {t.gap.corr(t.ict):.2f}, Spearman {t.gap.corr(t.ict, method='spearman'):.2f}\n", t.round(2).to_string())
    t = t.drop(index="B")  # mining: one industry, 0.2% of hours, far off the scale (+5.8)
    print(f"   without mining: Pearson {t.gap.corr(t.ict):.2f}, Spearman {t.gap.corr(t.ict, method='spearman'):.2f}, {len(t)} sections")
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 7.0), sharex=True, sharey=True); size = 40 + 18 * t.n
    hyp = -0.25 * t.ict  # invented example: the lag grows by 0.25 points for each basis point of ICT relevance
    axes[0].scatter(t.ict, hyp, s=size, color=EU_C, zorder=3, alpha=0.85, edgecolor=SURFACE); axes[0].set_title("Se la tecnologia fosse il problema\n(esempio inventato)", loc="left", fontsize=12)
    axes[1].scatter(t.ict, t.gap, s=size, color=ITALY_C, zorder=3, alpha=0.85, edgecolor=SURFACE); axes[1].set_title("I dati dell'Italia", loc="left", fontsize=12)
    off = {"G": (6, 8), "A": (6, 2), "R-S": (6, 7), "C": (10, 6), "I": (6, -8), "F": (6, -2), "H": (8, 0), "J": (-6, -11), "K": (-4, 11)}
    for sec, r in t.iterrows(): axes[1].annotate(SEC_IT.get(sec, sec), (r.ict, r.gap), xytext=off.get(sec, (6, 4)), textcoords="offset points", fontsize=10, color=INK2, ha="right" if sec in ("K", "J") else "left", va="center")
    for ax in axes: ax.axhline(0, color=INK2, lw=0.8)
    axes[1].set_xlabel("Rilevanza ICT (punti base)"); fig.supylabel("Produttività Italia − mediana (%)", fontsize=11.5, x=0.0); fig.tight_layout()
    save(fig, "01_ritardo_e_tecnologia")


def fig01_torneria() -> None:
    """Invented example of the post: a small lathe shop, 320 hours, value added 16,000 then 17,600 euro."""
    rows = [("Ore lavorate", [320, 320], "{:.0f}"), ("Valore aggiunto (€)", [16000, 17600], "{:,.0f}"), ("Euro per ora lavorata", [50, 55], "{:.0f}")]
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 4.4)); fig.suptitle("Una torneria (numeri inventati)", x=0.02, ha="left", fontsize=13, fontweight="bold")
    for ax, (lab, v, f) in zip(axes, rows, strict=True):
        ax.barh([0, 1], v, color=[PEER_C, ITALY_C], height=0.7); ax.set_yticks([0, 1], ["Anno 1", "Anno 2"]); ax.invert_yaxis(); ax.set_xlim(0, max(v) * 1.3); ax.set_xticks([]); ax.set_title(lab, loc="left", fontsize=12)
        for i, x in enumerate(v): ax.annotate(f.format(x).replace(",", " "), (x, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.grid(False)
    axes[2].annotate("+10%", (55, 1), xytext=(52, 1), fontsize=12, color="white", fontweight="bold", va="center", ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "01_torneria")


def fig01_tfp() -> None:
    """TFP contribution to labour productivity growth 1996-2019 (EU KLEMS), Italy and the median of the other members: market economy and four industries."""
    ek = pd.read_parquet(PROCESSED / "euklems_growth.parquet")
    def avg(n: str) -> pd.Series:
        d = ek[(ek.nace_r2_code == n) & (ek.variable == "LP1ConTFP") & ek.year.between(1996, 2019)]; return d.groupby("iso3").value.mean()
    rows = [("Economia di mercato", "MARKT"), ("Manifattura", "C"), ("Costruzioni", "F"), ("Servizi professionali\ne amministrativi", "M-N"), ("Energia, acqua, rifiuti", "D-E")]
    vals = {}
    for lab, n in rows:
        a = avg(n); o = a[[c for c in a.index if c in EU27_ISO3 and c != ITALY]].dropna(); vals[n] = (a[ITALY], o.median(), len(o))
    print("01 TFP 1996-2019 (pp a year), Italy / median / n others:", {k: tuple(round(float(x), 2) for x in v) for k, v in vals.items()})
    fig, ax = plt.subplots(figsize=(5.4, 4.2)); yy = np.arange(len(rows)); lo, hi = min(min(v[:2]) for v in vals.values()), max(max(v[:2]) for v in vals.values()); xl = hi + 1.0; xr = xl + 1.7
    ax.set_xlim(lo - 0.3, xr + 0.2); ax.set_ylim(len(rows) - 0.5, -1.3)
    for y, (_, n) in zip(yy, rows, strict=True):
        it, md, _ = vals[n]; ax.barh(y, it, color=ITALY_C, height=0.5, zorder=2); ax.scatter([md], [y], color=EU_C, marker="D", s=46, zorder=3)
        ax.annotate(f"{it:+.2f}".replace("-0.00", "0.00"), (xl, y), color=ITALY_C, fontweight="bold", fontsize=11.5, ha="right", va="center"); ax.annotate(f"{md:+.2f}", (xr, y), color=EU_C, fontweight="bold", fontsize=11.5, ha="right", va="center")
    ax.annotate("Italia", (xl, -0.9), color=ITALY_C, fontweight="bold", fontsize=11.5, ha="right", va="center"); ax.annotate("Mediana", (xr + 0.1, -0.9), color=EU_C, fontweight="bold", fontsize=11.5, ha="right", va="center")
    ax.set_yticks(yy, [lab for lab, _ in rows]); ax.axvline(0, color=INK2, lw=0.8); ax.set_xticks(range(-4, 2))
    ax.set_title("Produttività totale dei fattori,\ncontributo medio annuo 1996-2019", loc="left", fontsize=12); ax.set_xlabel("punti % l'anno (barra: Italia, rombo: mediana)")
    save(fig, "01_tfp")


def cover01(s: pd.DataFrame, others: list[str]) -> None:
    """Share/preview image 1200x630: the two numbers of the post (within-sector productivity growth, Italy and the median of the others)."""
    dec = pd.DataFrame({c: shift_share(s, c, SECT_A, SECT_B) for c in [ITALY, *others]}).T["within-sector productivity"]; it, eu = dec[ITALY], dec.drop(ITALY).median()
    print(f"01 cover: Italy {it:+.1f}%, median of the other {len(dec) - 1} {eu:+.1f}% (within-sector productivity per hour {SECT_A}-{SECT_B})")
    bg, fg, dim = "#14181f", "#ffffff", "#9aa3b2"
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=bg); ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 12); ax.set_ylim(0, 6.3); ax.axis("off")
    ax.text(6, 5.55, "Un problema di produttività", color=fg, fontsize=40, fontweight="bold", ha="center", va="center")
    ax.text(6, 4.85, f"Produttività oraria dentro i settori, {SECT_A} - {SECT_B}", color=dim, fontsize=24, ha="center", va="center")
    for y, val, col, who in [(3.45, it, ITALY_C, "Italia"), (1.45, eu, EU_C, "Mediana UE")]:
        ax.text(0.8, y, f"{val:+.0f}%".replace("-", "−"), color=col, fontsize=88, fontweight="bold", ha="left", va="center")
        ax.text(5.4, y + 0.42, who, color=col, fontsize=44, fontweight="bold", ha="left", va="center")
        ax.barh(y - 0.38, 5.6 * val / eu, left=5.4, height=0.5, color=col)
    ax.text(11.0, 0.3, "Cifre significative", color=dim, fontsize=15, ha="right", va="center")
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / "01_copertina.png", dpi=100, facecolor=bg); plt.close(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    apply_style(); apply_mobile_style(); style.FIG_DIR = OUT
    only = set(sys.argv[1:])
    panel, lh = load_panel(), load_long()
    if not only or "00" in only:
        fig00_decomposition(panel); fig00_long(lh); fig00_synth(lh); fig00_governments(lh); cover00(panel)
    if not only or "01" in only:
        sec = load_sectors(); sample = _sector_sample(sec); oth = [c for c in sample if c != ITALY]
        print(f"01 sample: {len(sample)} members with all nine sections in {SECT_A} and {SECT_B}; missing: {sorted(set(EU27_ISO3) - set(sample))}")
        fig01_index(sec, oth); fig01_shift(sec, oth); fig01_sectors(sec, oth); fig01_industries(); fig01_myths(); fig01_ict(); fig01_torneria(); fig01_tfp(); cover01(sec, oth)
