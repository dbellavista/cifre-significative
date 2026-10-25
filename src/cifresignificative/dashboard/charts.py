"""Plotly figure builders for the dashboard (pure functions: data in, figure out).

Colours follow the project palette: Italy blue, EU median orange dashed, other countries neutral gray; highlighted
countries take the next categorical slots in fixed order. Every line chart has a legend, hover tooltips and (in the app) a table view.
"""
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cifresignificative.analysis.data import EU27_ISO3, ITALY, eu_median_ex_italy

_THEMES: dict[str, dict[str, Any]] = {  # light and dark steps of the project palette (dark steps are the validated dark-surface variants)
    "light": {"italy": "#2a78d6", "eu": "#eb6834", "peer": "#b9b8b2", "highlight": ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
              "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e5e1", "surface": "#fcfcfb", "mid": "#f0efec", "lo": "#e34948", "hi": "#2a78d6", "debt_other": "#b9b8b2"},
    "dark": {"italy": "#3987e5", "eu": "#d95926", "peer": "#6b6a66", "highlight": ["#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
             "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#383835", "surface": "#1a1a19", "mid": "#383835", "lo": "#e66767", "hi": "#3987e5", "debt_other": "#6b6a66"},
}
ITALY_C = EU_C = PEER_C = INK = INK2 = GRID = SURFACE = MID = LO_C = HI_C = DEBT_OTHER = ""
HIGHLIGHT_C: list[str] = []
MAX_HIGHLIGHTS = 6
LAYOUT: dict = {}


def apply_theme(mode: str = "light") -> None:
    """Switch the module's colours (read at call time by every figure builder) between the light and dark palettes."""
    global ITALY_C, EU_C, PEER_C, HIGHLIGHT_C, INK, INK2, GRID, SURFACE, MID, LO_C, HI_C, DEBT_OTHER, LAYOUT
    t = _THEMES["dark" if mode == "dark" else "light"]
    ITALY_C, EU_C, PEER_C, HIGHLIGHT_C = t["italy"], t["eu"], t["peer"], t["highlight"]
    INK, INK2, GRID, SURFACE, MID, LO_C, HI_C, DEBT_OTHER = t["ink"], t["ink2"], t["grid"], t["surface"], t["mid"], t["lo"], t["hi"], t["debt_other"]
    LAYOUT = {"paper_bgcolor": SURFACE, "plot_bgcolor": SURFACE, "font": {"color": INK2, "size": 12}, "margin": {"l": 10, "r": 10, "t": 50, "b": 10},
              "xaxis": {"gridcolor": GRID, "linecolor": GRID, "zeroline": False}, "yaxis": {"gridcolor": GRID, "linecolor": GRID, "zeroline": False},
              "legend": {"orientation": "h", "y": -0.15}, "hoverlabel": {"bgcolor": SURFACE, "font": {"color": INK}}}


apply_theme("light")


def _style(fig: go.Figure, title: str, ylabel: str = "", height: int = 460) -> go.Figure:
    fig.update_layout(**LAYOUT, title={"text": title, "x": 0, "font": {"color": INK, "size": 16}}, height=height)
    fig.update_yaxes(title_text=ylabel)
    return fig


def line_fig(w: pd.DataFrame, title: str, ylabel: str = "", highlights: list[str] | None = None, show_peers: bool = True,
             benchmarks: list[str] | None = None, years: tuple[int, int] | None = None, rebase: bool = False, median: pd.Series | None = None,
             median_label: str = "Median of other EU27") -> go.Figure:
    """Italy vs median of the other EU27 vs optional highlighted countries; other EU27 members as gray context lines."""
    w = w.copy()
    if years:
        w = w.loc[years[0]:years[1]]
    if rebase:
        first = w.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
        w = w / first * 100
    highlights = [h for h in (highlights or []) if h != ITALY][:MAX_HIGHLIGHTS]
    med = (eu_median_ex_italy(w) if median is None else median.loc[w.index.min():w.index.max()]).dropna()
    if rebase and median is not None and med.notna().any():
        med = med / med.dropna().iloc[0] * 100
    fig = go.Figure()
    if show_peers:
        for c in EU27_ISO3:
            if c in w and c != ITALY and c not in highlights and w[c].notna().any():
                fig.add_trace(go.Scatter(x=w.index, y=w[c], mode="lines", line={"color": PEER_C, "width": 1}, name=c, legendgroup="peers",
                                         showlegend=False, hovertemplate=f"{c}: " + "%{y:.1f}<extra></extra>", opacity=0.7))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line={"color": PEER_C, "width": 1.5}, name="Other EU27", legendgroup="peers"))
    for c in benchmarks or []:
        if c in w:
            fig.add_trace(go.Scatter(x=w.index, y=w[c], mode="lines", line={"color": INK2, "width": 1.5, "dash": "dot"}, name=c))
    for c, col in zip(highlights, HIGHLIGHT_C, strict=False):
        if c in w and w[c].notna().any():
            fig.add_trace(go.Scatter(x=w.index, y=w[c], mode="lines+markers", line={"color": col, "width": 2.2}, marker={"size": 4}, name=c))
    fig.add_trace(go.Scatter(x=med.index, y=med.values, mode="lines", line={"color": EU_C, "width": 2.6, "dash": "dash"}, name=median_label))
    if ITALY in w:
        fig.add_trace(go.Scatter(x=w.index, y=w[ITALY], mode="lines+markers", line={"color": ITALY_C, "width": 3.4}, marker={"size": 5}, name="Italy"))
    fig.update_layout(hovermode="x unified")
    if rebase:
        ylabel = (ylabel + " " if ylabel else "") + "(first year = 100)"
    return _style(fig, title, ylabel)


def multiples_fig(items: list[tuple[pd.DataFrame, str]], ncols: int = 3, years: tuple[int, int] | None = None) -> go.Figure:
    """Small multiples: items = [(year x iso3 table, title)]."""
    nrows = int(np.ceil(len(items) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[t for _, t in items], horizontal_spacing=0.06, vertical_spacing=0.16)
    for k, (w, _) in enumerate(items):
        r, c = divmod(k, ncols)
        if years:
            w = w.loc[years[0]:years[1]]
        med = eu_median_ex_italy(w).dropna()
        for cc in EU27_ISO3:
            if cc in w and cc != ITALY and w[cc].notna().any():
                fig.add_trace(go.Scatter(x=w.index, y=w[cc], mode="lines", line={"color": PEER_C, "width": 0.8}, opacity=0.6, showlegend=False,
                                         hovertemplate=f"{cc}: " + "%{y:.1f}<extra></extra>"), row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=med.index, y=med.values, mode="lines", line={"color": EU_C, "width": 2, "dash": "dash"}, name="Median of other EU27",
                                 showlegend=(k == 0), legendgroup="m", hovertemplate="EU median: %{y:.1f}<extra></extra>"), row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=w.index, y=w[ITALY], mode="lines", line={"color": ITALY_C, "width": 3}, name="Italy", showlegend=(k == 0), legendgroup="i",
                                 hovertemplate="Italy: %{y:.1f}<extra></extra>"), row=r + 1, col=c + 1)
    fig.update_layout(**{**LAYOUT, "xaxis": None, "yaxis": None}, height=300 * nrows + 60, hovermode="closest")
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_annotations(font={"color": INK, "size": 12})
    return fig


def gap_heatmap_fig(gaps: pd.DataFrame, title: str) -> go.Figure:
    """gaps: columns group, label, italy, median_others, z_gap. Diverging blue(higher) - gray - red(lower)."""
    labels = [f"{g} | {l}" for g, l in zip(gaps.group, gaps.label, strict=True)]
    custom = np.c_[gaps.italy.round(1), gaps.median_others.round(1)]
    fig = go.Figure(go.Heatmap(z=gaps.z_gap.clip(-3, 3).values[:, None], y=labels, x=["Italy vs median"], zmin=-3, zmax=3,
                               colorscale=[[0, LO_C], [0.5, MID], [1, HI_C]], customdata=custom,
                               hovertemplate="%{y}<br>Italy %{customdata[0]} vs median %{customdata[1]}<br>gap %{z:+.1f} SD<extra></extra>",
                               colorbar={"title": "SD", "tickvals": [-3, -2, -1, 0, 1, 2, 3]}))
    fig.update_yaxes(autorange="reversed", gridcolor=GRID)
    fig = _style(fig, title, height=max(420, 26 * len(gaps) + 120))
    fig.update_layout(margin={"l": 10, "r": 10, "t": 60, "b": 10})
    for i, z in enumerate(gaps.z_gap):
        fig.add_annotation(x=0, y=labels[i], text=f"{z:+.1f}", showarrow=False, font={"size": 11, "color": INK})
    return fig


def scatter_fig(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, title: str) -> go.Figure:
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    others = d.drop(ITALY, errors="ignore")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=others.x, y=others.y, mode="markers+text", text=others.index, textposition="top center", textfont={"size": 9, "color": INK2},
                             marker={"color": PEER_C, "size": 9, "line": {"color": SURFACE, "width": 1}}, name="Other EU27",
                             hovertemplate="%{text}<br>" + f"{xlabel}: " + "%{x:.1f}<br>" + f"{ylabel}: " + "%{y:.1f}<extra></extra>"))
    if len(others) > 3:
        b = np.polyfit(others.x, others.y, 1)
        xs = np.linspace(others.x.min(), others.x.max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines", line={"color": EU_C, "dash": "dash", "width": 2}, name="Fit, other EU27", hoverinfo="skip"))
    if ITALY in d.index:
        fig.add_trace(go.Scatter(x=[d.loc[ITALY, "x"]], y=[d.loc[ITALY, "y"]], mode="markers+text", text=["Italy"], textposition="top center",
                                 textfont={"color": ITALY_C, "size": 13}, marker={"color": ITALY_C, "size": 15, "line": {"color": SURFACE, "width": 2}}, name="Italy",
                                 hovertemplate="Italy<br>" + f"{xlabel}: " + "%{x:.1f}<br>" + f"{ylabel}: " + "%{y:.1f}<extra></extra>"))
    fig.update_xaxes(title_text=xlabel)
    return _style(fig, title, ylabel)


def region_scatter_fig(d: pd.DataFrame, xlabel: str, ylabel: str, title: str, areas: dict[str, str]) -> go.Figure:
    """d: index = NUTS2 code, columns x, y. Other regions gray with a fit line; Italian regions coloured by macro-area (`areas` maps code -> area)."""
    d = d.dropna()
    it = d[d.index.str.startswith("IT")]
    others = d.drop(it.index)
    hover = "%{text}<br>" + f"{xlabel}: " + "%{x:.2f}<br>" + f"{ylabel}: " + "%{y:.2f}<extra></extra>"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=others.x, y=others.y, mode="markers", text=others.index, marker={"color": PEER_C, "size": 7, "opacity": 0.7}, name="Other regions", hovertemplate=hover))
    if len(others) > 3:
        b = np.polyfit(others.x, others.y, 1)
        xs = np.linspace(d.x.min(), d.x.max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines", line={"color": EU_C, "dash": "dash", "width": 2}, name="Fit, non-Italian regions", hoverinfo="skip"))
    order = ["North-West", "North-East", "Centre", "South", "Islands"]
    colours = [ITALY_C, HIGHLIGHT_C[0], HIGHLIGHT_C[1], HIGHLIGHT_C[2], HIGHLIGHT_C[3]]
    for area, colour in zip(order, colours):
        sub = it[[areas.get(g, "") == area for g in it.index]]
        if len(sub):
            fig.add_trace(go.Scatter(x=sub.x, y=sub.y, mode="markers", text=sub.index, name=f"Italy: {area}", hovertemplate=hover,
                                     marker={"color": colour, "size": 11, "line": {"color": SURFACE, "width": 1.5}}))
    fig.update_xaxes(title_text=xlabel)
    return _style(fig, title, ylabel)


def events_line_fig(s: pd.Series, events: pd.DataFrame, title: str, ylabel: str) -> go.Figure:
    """One monthly series with dated events as vertical lines: governments (dotted gray), ECB decisions (dashed orange), policies (thin blue)."""
    s = s.dropna()
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line={"color": ITALY_C, "width": 2}, name=ylabel or "series", hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"))
    style = {"government": (PEER_C, "dot", "Italian government"), "ecb": (EU_C, "dash", "ECB decision"), "policy": (HIGHLIGHT_C[0], "solid", "Italian policy")}
    for kind, (colour, dash, name) in style.items():
        sub = events[events.type == kind]
        sub = sub[(sub.date >= s.index.min()) & (sub.date <= s.index.max())]
        for d in sub.date:
            fig.add_shape(type="line", x0=d, x1=d, y0=0, y1=1, yref="paper", line={"color": colour, "width": 1.5 if kind == "ecb" else 0.8, "dash": dash}, opacity=0.8 if kind != "policy" else 0.35)
        if len(sub):
            fig.add_trace(go.Scatter(x=sub.date, y=[s.reindex(s.index.union(sub.date)).ffill().get(d, np.nan) for d in sub.date], mode="markers", marker={"color": colour, "size": 7}, text=sub.event,
                                     name=name, hovertemplate="%{text}<br>%{x|%d %b %Y}<extra></extra>"))
    return _style(fig, title, ylabel)


def bars_fig(df: pd.DataFrame, cols: dict[str, str], title: str, xlabel: str = "", horizontal: bool = True, stacked: bool = False) -> go.Figure:
    """df index = categories; cols = {column: colour}."""
    fig = go.Figure()
    for col, colour in cols.items():
        if horizontal:
            fig.add_trace(go.Bar(y=df.index, x=df[col], orientation="h", name=col, marker={"color": colour}, hovertemplate="%{y}: %{x:.2f}<extra>" + col + "</extra>"))
        else:
            fig.add_trace(go.Bar(x=df.index, y=df[col], name=col, marker={"color": colour}, hovertemplate="%{x}: %{y:.2f}<extra>" + col + "</extra>"))
    fig.update_layout(barmode="relative" if stacked else "group", bargap=0.35)
    if horizontal:
        fig.update_xaxes(title_text=xlabel)
        fig.update_yaxes(autorange="reversed")
    return _style(fig, title, height=max(380, 30 * len(df) + 140) if horizontal else 460)


def signed_bars_fig(s: pd.Series, title: str, xlabel: str) -> go.Figure:
    """Single series, blue where Italy is above the median, orange where below (colour + sign + value label)."""
    s = s.sort_values()
    fig = go.Figure(go.Bar(y=s.index, x=s.values, orientation="h", marker={"color": [ITALY_C if v >= 0 else EU_C for v in s.values]},
                           text=[f"{v:+.1f}" for v in s.values], textposition="outside", hovertemplate="%{y}: %{x:+.2f}<extra></extra>", showlegend=False))
    fig.add_vline(x=0, line_color=INK2, line_width=1)
    fig.update_xaxes(title_text=xlabel)
    return _style(fig, title, height=max(380, 30 * len(s) + 140))


def strip_fig(df: pd.DataFrame, title: str, ylabel: str) -> go.Figure:
    """df columns: country, region, value, italy(bool). One dot per region, countries on the x axis (sorted by median)."""
    order = df.groupby("country").value.median().sort_values().index.tolist()
    fig = go.Figure()
    for is_it, name, col, size in [(False, "Other EU27 regions", PEER_C, 8), (True, "Italian regions", ITALY_C, 11)]:
        d = df[df.italy == is_it]
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(d))
        pos = d.country.map({c: i for i, c in enumerate(order)}) + jitter
        fig.add_trace(go.Scatter(x=pos, y=d.value, mode="markers", name=name, marker={"color": col, "size": size, "line": {"color": SURFACE, "width": 1}},
                                 text=d.region, customdata=d.country, hovertemplate="%{text} (%{customdata})<br>%{y:,.1f}<extra></extra>"))
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(order))), ticktext=order)
    return _style(fig, title, ylabel)



def debt_fig(dec: pd.DataFrame, title: str) -> go.Figure:
    """dec: rows = countries, columns = DEBT_COLORS keys (pp of GDP). Stacked contributions with a marker for the total change."""
    colors = {"primary balance": ITALY_C, "interest": LO_C, "nominal growth": HIGHLIGHT_C[0], "other (stock-flow)": DEBT_OTHER}
    fig = go.Figure()
    for col, colour in colors.items():
        fig.add_trace(go.Bar(x=dec.index, y=dec[col], name=col, marker={"color": colour}, hovertemplate="%{x}: %{y:+.1f} pp<extra>" + col + "</extra>"))
    fig.add_trace(go.Scatter(x=dec.index, y=dec.sum(axis=1), mode="markers", name="change in debt/GDP", marker={"color": INK, "size": 9},
                             hovertemplate="%{x}: %{y:+.1f} pp<extra>total</extra>"))
    fig.update_layout(barmode="relative", bargap=0.35)
    return _style(fig, title, "pp of GDP")


def timeline_fig(italy: pd.Series, median: pd.Series | None, governments: pd.DataFrame, policies: pd.DataFrame, title: str, ylabel: str, xlim: tuple[int, int]) -> go.Figure:
    """Italy (and optionally the EU median) with government terms as alternating bands and policy dates as markers (hover shows the law)."""
    fig = go.Figure()
    if median is not None:
        m = median.dropna()
        fig.add_trace(go.Scatter(x=m.index, y=m.values, mode="lines", line={"color": EU_C, "width": 2.4, "dash": "dash"}, name="Median of other EU27"))
    i = italy.dropna()
    fig.add_trace(go.Scatter(x=i.index, y=i.values, mode="lines+markers", line={"color": ITALY_C, "width": 3.2}, marker={"size": 5}, name="Italy"))
    for k, g in enumerate(governments.to_dict("records")):
        x0 = g["start"].year + (g["start"].dayofyear - 1) / 365.25
        x1 = g["end_filled"].year + (g["end_filled"].dayofyear - 1) / 365.25
        if x1 < xlim[0] or x0 > xlim[1] + 1:
            continue
        fig.add_vrect(x0=max(x0, xlim[0]), x1=min(x1, xlim[1] + 1), fillcolor=INK, opacity=0.05 if k % 2 else 0.0, line_width=0, layer="below",
                      annotation_text=g["government"] if (x1 - x0) > 0.4 else "", annotation_position="top", annotation_textangle=-90, annotation_font={"size": 9, "color": INK2})
    ymin = float(min(i.min(), m.min() if median is not None and len(m) else i.min()))
    ymax = float(max(i.max(), m.max() if median is not None and len(m) else i.max()))
    base = ymin - 0.06 * (ymax - ymin)
    pol = policies.assign(x=policies.date.dt.year + (policies.date.dt.dayofyear - 1) / 365.25)
    pol = pol[(pol.x >= xlim[0]) & (pol.x <= xlim[1] + 1)]
    for ev, sym in [("verified", "diamond"), ("partly verified", "diamond-open"), ("unverified", "diamond-open")]:
        d = pol[pol.evidence == ev]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(x=d.x, y=[base] * len(d), mode="markers", marker={"symbol": sym, "size": 11, "color": INK2, "line": {"color": INK2, "width": 1.5}}, name=f"policy ({ev})",
                                 text=[f"{t.date:%Y-%m-%d}: {t.title}<br>{t.reference}<br>{t.what_it_changed}" for t in d.itertuples()], hovertemplate="%{text}<extra></extra>"))
    fig.update_xaxes(range=[xlim[0], xlim[1] + 1])
    fig.update_layout(hovermode="closest")
    return _style(fig, title, ylabel, 520)
