"""Chart helpers: Italy in blue, EU median in orange (dashed), peers in neutral gray; one axis per chart.

Colors come from the validated reference palette in the dataviz skill (categorical slots 1 and 2, diverging
blue<->red with a gray midpoint). Italy vs the EU is a two-series story, so every chart keeps identity through
color *and* line style / direct labels, never color alone.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from cifresignificative.studies.italy_productivity.analysis.data import (
    BENCHMARKS,
    EU27_ISO3,
    ITALY,
    eu_median_ex_italy,
    wide,
)

ITALY_C = "#2a78d6"
EU_C = "#eb6834"
PEER_C = "#b9b8b2"
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#e34948", "#f0efec", "#2a78d6"])  # low=red, high=blue
FIG_DIR = Path("studies/italy_productivity/reports/figures")


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110, "lines.solid_capstyle": "round",
    })


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=160, bbox_inches="tight")


def line_panel(ax, panel: pd.DataFrame, indicator: str, title: str | None = None, ylabel: str = "", peers=None,
               show_benchmarks: bool = False, rebase_to: int | None = None, label_end: bool = True):
    """Italy (blue) vs median of other EU27 (orange dashed) over gray peer lines."""
    w = wide(panel, indicator)
    if rebase_to is not None:
        w = w / w.loc[rebase_to] * 100
    med = eu_median_ex_italy(w)
    gray = [*(peers if peers is not None else [c for c in EU27_ISO3 if c != ITALY]), *(BENCHMARKS if show_benchmarks else [])]
    for c in gray:
        if c in w:
            ax.plot(w.index, w[c], color=PEER_C, lw=0.8, alpha=0.8, zorder=1)
    ax.plot(med.index, med.values, color=EU_C, lw=2, ls="--", zorder=2)
    ax.plot(w.index, w[ITALY], color=ITALY_C, lw=2.4, zorder=3)
    if label_end:
        x = w[ITALY].dropna().index.max()
        xm = med.dropna().index.max()
        up = w.loc[x, ITALY] >= med[xm]  # whichever line is higher gets the upper label, so the two never collide
        ax.annotate("Italy", (x, w.loc[x, ITALY]), xytext=(4, 6 if up else -6), textcoords="offset points", color=ITALY_C, fontsize=9, va="center", fontweight="bold")
        ax.annotate("EU median", (xm, med[xm]), xytext=(4, -6 if up else 6), textcoords="offset points", color=EU_C, fontsize=8, va="center")
    pooled = pd.concat([w[c] for c in gray if c in w] + [w[ITALY], med]).dropna()
    lo, hi = pooled.quantile(0.01), pooled.quantile(0.99)
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)  # outliers (e.g. Luxembourg, Ireland) run off the frame instead of flattening the rest
    ax.set_title(title or indicator)
    ax.set_ylabel(ylabel)
    ax.margins(x=0.02)
    return ax


def line_wide(ax, w: pd.DataFrame, title: str, ylabel: str = "", label_end: bool = True):
    """Same Italy / EU-median / gray-peers chart for any year x iso3 table (sparse series get markers)."""
    med = eu_median_ex_italy(w)
    mk = {"marker": "o", "ms": 4} if w[ITALY].notna().sum() < 9 else {}
    for c in [c for c in EU27_ISO3 if c != ITALY and c in w]:
        d = w[c].dropna()
        ax.plot(d.index, d.values, color=PEER_C, lw=0.8, alpha=0.8, zorder=1)
    m = med.dropna(); i = w[ITALY].dropna()
    ax.plot(m.index, m.values, color=EU_C, lw=2, ls="--", zorder=2, **mk)
    ax.plot(i.index, i.values, color=ITALY_C, lw=2.4, zorder=3, **mk)
    if label_end and len(i):
        up = i.iloc[-1] >= m.iloc[-1]
        ax.annotate("Italy", (i.index[-1], i.iloc[-1]), xytext=(4, 6 if up else -6), textcoords="offset points", color=ITALY_C, fontsize=9, va="center", fontweight="bold")
        ax.annotate("EU median", (m.index[-1], m.iloc[-1]), xytext=(4, -6 if up else 6), textcoords="offset points", color=EU_C, fontsize=8, va="center")
    pooled = pd.concat([w[c].dropna() for c in w if c in EU27_ISO3]).dropna()
    lo, hi = pooled.quantile(0.01), pooled.quantile(0.99); pad = 0.06 * (hi - lo)
    ax.set_ylim(min(lo, i.min() if len(i) else lo) - pad, max(hi, i.max() if len(i) else hi) + pad)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.margins(x=0.02)
    return ax


def small_multiples_wide(items: list[tuple[pd.DataFrame, str, str]], ncols: int = 3, name: str | None = None):
    """items: (year x iso3 table, title, ylabel). Same legend as small_multiples."""
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    for ax, (w, t, yl) in zip(axes.flat, items, strict=False):
        line_wide(ax, w, t, yl)
    for ax in list(axes.flat)[len(items):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=ITALY_C, lw=2.4, label="Italy"), plt.Line2D([], [], color=EU_C, lw=2, ls="--", label="Median of other EU27"),
               plt.Line2D([], [], color=PEER_C, lw=1, label="Other EU27 countries")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    if name:
        save(fig, name)
    return fig


def small_multiples(panel: pd.DataFrame, specs: list[tuple[str, str, str]], ncols: int = 3, name: str | None = None, **kw):
    """specs: (indicator, title, ylabel). Returns the figure."""
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    for ax, (ind, title, yl) in zip(axes.flat, specs, strict=False):
        line_panel(ax, panel, ind, title, yl, **kw)
    for ax in list(axes.flat)[len(specs):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=ITALY_C, lw=2.4, label="Italy"), plt.Line2D([], [], color=EU_C, lw=2, ls="--", label="Median of other EU27"),
               plt.Line2D([], [], color=PEER_C, lw=1, label="Other EU27 countries")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    if name:
        save(fig, name)
    return fig


def gap_heatmap(gaps: pd.DataFrame, title: str, name: str | None = None, vlim: float = 3.0):
    """gaps: rows = indicators (labels), columns = periods, values = z-gap (Italy vs EU median in cross-country SD)."""
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * gaps.shape[1], 0.34 * len(gaps) + 1.4))
    im = ax.imshow(gaps.values, cmap=DIVERGING, vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(gaps.shape[1]), gaps.columns)
    ax.set_yticks(range(len(gaps)), gaps.index)
    ax.grid(False)
    ax.xaxis.tick_top()
    for i in range(gaps.shape[0]):
        for j in range(gaps.shape[1]):
            v = gaps.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Italy minus median of other EU27, in standard deviations\n(red = Italy lower, blue = Italy higher)", fontsize=8)
    ax.set_title(title, pad=24)
    for s in ax.spines.values():
        s.set_visible(False)
    if name:
        save(fig, name)
    return fig


def scatter_italy(ax, x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, title: str, label_peers=True):
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    others = d.drop(ITALY, errors="ignore")
    ax.scatter(others.x, others.y, s=34, color=PEER_C, edgecolor=SURFACE, linewidth=1, zorder=2)
    if len(others) > 3:
        b = np.polyfit(others.x, others.y, 1)
        xs = np.linspace(others.x.min(), others.x.max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=EU_C, lw=1.6, ls="--", zorder=1)
    if label_peers:
        for c, r in others.iterrows():
            if c in ("DEU", "FRA", "ESP", "GRC", "POL", "PRT", "NLD", "SWE", "IRL", "LUX"):
                ax.annotate(c, (r.x, r.y), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
    if ITALY in d.index:
        ax.scatter(d.loc[ITALY, "x"], d.loc[ITALY, "y"], s=90, color=ITALY_C, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.annotate("Italy", (d.loc[ITALY, "x"], d.loc[ITALY, "y"]), xytext=(6, 5), textcoords="offset points", color=ITALY_C, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def draw_timeline(ax, governments: pd.DataFrame, policies: pd.DataFrame | None = None, xlim: tuple[int, int] = (2005, 2025), label_every_government: bool = True, policy_labels: bool = False):
    """Shade each government's term (alternating light bands, name on top) and mark policy dates (diamonds on the baseline of the axis)."""
    lo, hi = pd.Timestamp(xlim[0], 1, 1), pd.Timestamp(xlim[1], 12, 31)
    ymin, ymax = ax.get_ylim()
    for k, g in enumerate(governments.to_dict("records")):
        s, e = max(g["start"], lo), min(g["end_filled"], hi)
        if e <= s:
            continue
        ax.axvspan(s.year + (s.dayofyear - 1) / 365.25, e.year + (e.dayofyear - 1) / 365.25, color="#000000", alpha=0.05 if k % 2 else 0.0, lw=0)
        if label_every_government and (e - s).days > 150:
            ax.text((s.year + (s.dayofyear - 1) / 365.25 + e.year + (e.dayofyear - 1) / 365.25) / 2, ymax, g["government"], rotation=90, ha="center", va="top", fontsize=7, color=INK2)
    if policies is not None and len(policies):
        yy = ymin + 0.03 * (ymax - ymin)
        for r in policies.to_dict("records"):
            x = r["date"].year + (r["date"].dayofyear - 1) / 365.25
            if not xlim[0] <= x <= xlim[1] + 1:
                continue
            ax.plot([x, x], [ymin, ymax], color=INK2, lw=0.6, ls=":", zorder=1)
            ax.scatter([x], [yy], marker="D", s=26, color=INK2 if r["evidence"] == "verified" else "none", edgecolor=INK2, zorder=4)
            if policy_labels:
                ax.annotate(r["title"][:34], (x, yy), xytext=(3, 6), textcoords="offset points", fontsize=6.5, rotation=90, color=INK2, va="bottom")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xlim[0], xlim[1] + 1)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))  # integer years, not 2007.5
    return ax
