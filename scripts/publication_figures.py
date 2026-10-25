"""Figures for the Italian series of posts, labels in Italian, same palette as the notebooks.

Usage: uv run python scripts/publication_figures.py [00 ...]   -> figures/NN_name.png; the numbers quoted in
the post are printed so they can be re-checked against the notebooks before publishing.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cifresignificative.analysis.data import (
    ITALY,
    gdp_pc_growth_decomposition,
    load_long,
    load_panel,
    long_wide,
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
END = 2025  # last year of the post: 2025 is a first estimate, said in the text
NAMES = {"DEU": "Germania", "FRA": "Francia", "ESP": "Spagna", "GBR": "Regno Unito", "PRT": "Portogallo", "GRC": "Grecia", "AUT": "Austria", "BEL": "Belgio", "DNK": "Danimarca", "FIN": "Finlandia", "IRL": "Irlanda", "LUX": "Lussemburgo", "NLD": "Paesi Bassi", "SWE": "Svezia", "USA": "Stati Uniti", "JPN": "Giappone"}


def save(fig, name: str, note: str = "", dy: float = -0.02) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if note:
        fig.text(0.0, dy, note, fontsize=7.5, color=INK2, ha="left", va="top")
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
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
    labels = ["Tasso di occupazione", "Ore per lavoratore", "Produttività e demografia (residuo)"]
    show = dec.loc[["ITA", "DEU", "FRA", "ESP", "PRT", "GRC"]].copy()
    show.loc[f"Mediana degli altri {len(dec) - 1}"] = dec.drop("ITA").median()
    show.index = [NAMES.get(i, "Italia" if i == "ITA" else i) for i in show.index]
    print(f"00 decomposition 2005-{END} ({len(dec)} countries, log points x100):\n", show.round(1).to_string())
    print("   rank of Italy by total (1 = lowest):", int(dec["real GDP per head"].rank()["ITA"]), "| lowest:", dec["real GDP per head"].nsmallest(4).round(1).to_dict())
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    cols = ["#2a78d6", "#eda100", "#8a8a85"]
    pos, neg = np.zeros(len(show)), np.zeros(len(show))
    for part, lab, col in zip(parts, labels, cols, strict=True):
        v = show[part].values; base = np.where(v >= 0, pos, neg)
        ax.barh(range(len(show)), v, left=base, color=col, height=0.62, label=lab, edgecolor=SURFACE, linewidth=2)
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.scatter(show["real GDP per head"], range(len(show)), color=INK, zorder=5, s=28, label="PIL reale per abitante (totale)")
    for i, t in enumerate(show["real GDP per head"]): ax.annotate(f"{t:+.0f}", (t, i), xytext=(7, -11), textcoords="offset points", fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(range(len(show)), show.index); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel(f"variazione in punti logaritmici (x100), 2005-{END}")
    ax.set_title(f"Da dove viene la crescita del PIL per abitante, 2005-{END}", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10)
    save(fig, "00_decomposizione_crescita")


def fig00_long(lh) -> None:
    gdp = long_wide(lh, "gdp_real_per_head_ncu"); idx = gdp / gdp.loc[1995] * 100
    g = gdp.apply(np.log).diff() * 100
    dec = pd.DataFrame({f"{d}s": g.loc[d:d + 9].mean() for d in range(1960, 2020, 10)}); dec[f"2020-{END % 100}"] = g.loc[2020:END].mean()
    print("00 growth by decade (% a year):\n", dec.loc[[ITALY, "DEU", "FRA", "ESP"]].round(1).to_string())
    print(f"00 index 1995=100 in {END}:", idx.loc[END, [ITALY, "DEU", "FRA", "ESP", "GBR"]].round(1).to_dict())
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    peers = ["DEU", "FRA", "ESP", "GBR"]; ypos = spread(idx.loc[END, peers].to_dict(), 4)  # end labels at least 4 index points apart
    for c in peers:
        ax.plot(idx.index, idx[c], color=PEER_C, lw=1.5)
        ax.annotate(NAMES[c], (END, idx.loc[END, c]), xytext=(END + 6.5, ypos[c]), textcoords="data", fontsize=8, color=INK2, va="center", arrowprops={"arrowstyle": "-", "color": PEER_C, "lw": 0.6, "shrinkA": 0, "shrinkB": 0})
    ax.plot(idx.index, idx[ITALY], color=ITALY_C, lw=2.8)
    ax.annotate("Italia", (END, idx.loc[END, ITALY]), xytext=(4, 0), textcoords="offset points", color=ITALY_C, fontweight="bold", va="center")
    ax.axvline(1999, color=INK2, lw=0.8, ls=":"); ax.annotate("euro", (1999.5, 30), fontsize=8, color=INK2)
    ax.axhline(100, color=INK2, lw=0.6, ls=":"); ax.annotate("1995 = 100 per ogni paese", (1961, 103), fontsize=8, color=INK2)
    ax.set_title("Indice di crescita del PIL reale per abitante (1995 = 100)", loc="left"); ax.set_ylabel("indice di crescita (non è un livello di reddito)"); ax.set_xlim(1960, END + 11)
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
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.plot(yy.index, yy[ITALY], color=ITALY_C, lw=2.6); ax.plot(synth.index, synth, color=EU_C, lw=2.2, ls="--")
    ax.annotate("Italia", (END, yy.loc[END, ITALY]), xytext=(4, -6), textcoords="offset points", color=ITALY_C, fontweight="bold")
    ax.annotate("Italia sintetica", (END, synth.loc[END]), xytext=(4, 4), textcoords="offset points", color=EU_C, fontweight="bold")
    ax.axvline(1995, color=INK2, lw=0.8, ls=":"); ax.annotate("1995", (1995.4, -58), fontsize=8, color=INK2); ax.set_xlim(1970, END + 9)
    ax.set_title("PIL reale per abitante, Italia e Italia sintetica (punti log, 1995 = 0)", loc="left"); ax.set_ylabel("punti logaritmici")
    save(fig, "00_italia_sintetica")
    gaps: dict[str, pd.Series] = {}; pre_err: dict[str, float] = {}; post_err: dict[str, float] = {}  # every unit against its own mix of the others
    for u in units:
        _, s = synthetic_control(yy, u, pre, [c for c in units if c != u]); gaps[u] = yy[u] - s; pre_err[u] = rmspe(yy[u], s, pre); post_err[u] = rmspe(yy[u], s, (1995, END))
    cut = float(np.median(list(pre_err.values())))
    print(f"00 placebo split at the median pre-1995 error {cut:.2f}:")
    for u in sorted(units, key=lambda u: pre_err[u]): print(f"   {u} pre {pre_err[u]:.1f} post {post_err[u]:.1f} ratio {ratios[u]:.1f} gap {END} {gaps[u].loc[END]:.1f}")
    for name, title, members in [("00_placebo_ben_riprodotti", "Paesi riprodotti bene prima del 1995", [u for u in units if pre_err[u] <= cut]), ("00_placebo_mal_riprodotti", "Paesi riprodotti male prima del 1995", [u for u in units if pre_err[u] > cut])]:
        fig, ax = plt.subplots(figsize=(9.4, 4.8)); others = [u for u in members if u != ITALY]
        ends = spread({u: float(gaps[u].loc[END]) for u in others}, 4.5)
        for u in others:
            gp = gaps[u]; ax.plot(gp.index, gp, color=PEER_C, lw=1.3)
            ax.annotate(f"{NAMES.get(u, u)} ({pre_err[u]:.1f} → {post_err[u]:.1f})", (END, gp.loc[END]), xytext=(END + 2, ends[u]), textcoords="data", fontsize=8, color=INK2, va="center", arrowprops={"arrowstyle": "-", "color": PEER_C, "lw": 0.5, "shrinkA": 0, "shrinkB": 0})
        ax.plot(gap.index, gap, color=ITALY_C, lw=2.8); ax.annotate(f"Italia ({pre_err[ITALY]:.1f} → {post_err[ITALY]:.1f})", (END, gap.loc[END]), xytext=(END + 2, gap.loc[END]), color=ITALY_C, fontweight="bold", va="center", fontsize=9)
        ax.axhline(0, color=INK2, lw=0.8); ax.axvline(1995, color=INK2, lw=0.8, ls=":"); ax.set_xlim(1970, END + 17)
        ax.set_title(title, loc="left"); ax.set_ylabel("scarto dalla propria versione sintetica\n(punti logaritmici)")
        ax.annotate("tra parentesi: errore medio prima → dopo il 1995", (1971, ax.get_ylim()[0]), xytext=(0, 6), textcoords="offset points", fontsize=8, color=INK2)
        save(fig, name)


def fig00_governments(lh) -> None:
    stats = government_stats(lh, load_governments("1989-01-01"), "1989-01-01")
    st = stats.dropna(subset=["gdp_growth_pa_italy", "gdp_growth_pa_others"]).reset_index(drop=True)
    slower = int((st.gdp_growth_pa_italy < st.gdp_growth_pa_others).sum())
    print(f"00 governments: slower under {slower} of {len(st)}; faster under:", st[st.gdp_growth_pa_italy >= st.gdp_growth_pa_others].government.tolist())
    print(st[["government", "start", "months", "gdp_growth_pa_italy", "gdp_growth_pa_others"]].round(1).to_string())
    yy = np.arange(len(st))
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(st) + 1.6))
    ax.barh(yy - 0.18, st.gdp_growth_pa_italy, height=0.34, color=ITALY_C, label="Italia")
    ax.barh(yy + 0.18, st.gdp_growth_pa_others, height=0.34, color=EU_C, label="mediana degli altri paesi UE")
    ax.set_yticks(yy, [f"{g} ({s.year})" for g, s in zip(st.government, st.start, strict=True)], fontsize=8); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title("Crescita del PIL reale, % l'anno, durante ciascun governo", loc="left"); ax.legend(fontsize=8, loc="lower right"); ax.set_ylim(len(st) - 0.4, -0.7)
    save(fig, "00_crescita_per_governo")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    apply_style(); style.FIG_DIR = OUT
    only = set(sys.argv[1:])
    panel, lh = load_panel(), load_long()
    if not only or "00" in only:
        fig00_decomposition(panel); fig00_long(lh); fig00_synth(lh); fig00_governments(lh)
