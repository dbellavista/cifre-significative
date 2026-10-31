"""Cells of notebook 07_regional_italy."""
from . import NOTEBOOKS, C, M

NOTEBOOKS["07_regional_italy"] = [
    (M, "# 07 - Regional Italy (NUTS2)\nNorth-South dispersion in Italy compared with regional dispersion in other EU27 countries."),
    (C, '''reg = load_regional()
def rwide(ind): return reg[reg.indicator == ind].pivot(index="year", columns="geo", values="value")
gdp = rwide("nuts2_gdp_pps_pc_eu27"); itg = [g for g in gdp.columns if g.startswith("IT")]
last = gdp[itg].dropna(how="all").index.max()
area = lambda g: {"ITC": "North-West", "ITH": "North-East", "ITI": "Centre", "ITF": "South", "ITG": "Islands"}.get(g[:3], "other")
order = ["North-West", "North-East", "Centre", "South", "Islands"]
print("Last year with data:", last)
snap = pd.DataFrame({"gdp_pps_eur_per_head": gdp.loc[last, itg]}); snap["area"] = [area(g) for g in snap.index]
display(snap.groupby("area").gdp_pps_eur_per_head.agg(["min", "median", "max"]).loc[order].round(0))'''),
    (C, '''# Distribution of regional GDP per head (PPS, EUR per inhabitant) by country: one dot per NUTS2 region
cs = gdp.loc[last].dropna().rename("v").to_frame(); cs["c"] = cs.index.str[:2]
counts = cs.groupby("c").size(); cs = cs[cs.c.isin(counts[counts >= 3].index)]
orderc = cs.groupby("c").v.median().sort_values().index
fig, ax = plt.subplots(figsize=(9, 4.6)); rng = np.random.default_rng(0)
for i, c in enumerate(orderc):
    v = cs[cs.c == c].v; col = ITALY_C if c == "IT" else PEER_C
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=34 if c == "IT" else 22, color=col, edgecolor=SURFACE, linewidth=0.8, zorder=3 if c == "IT" else 2)
eu_med = gdp.loc[last].median(); ax.axhline(eu_med, color=EU_C, ls="--", lw=1.5); ax.annotate("median EU27 region", (0, eu_med), xytext=(0, 4), textcoords="offset points", fontsize=8, color=EU_C)
ax.set_xticks(range(len(orderc)), orderc); ax.set_ylabel("GDP per head, PPS (EUR)"); ax.set_title(f"Regional GDP per head by country, {last}"); save(fig, "07_regional_dispersion")
cv = cs.groupby("c").v.agg(lambda s: s.std() / s.mean()).sort_values(ascending=False)
print("Coefficient of variation across regions (higher = more unequal):", cv.head(6).round(2).to_dict(), "| Italy rank:", list(cv.index).index("IT") + 1, "of", len(cv))'''),
    (C, '''# Labour and education gaps inside Italy, first vs last available year
def italian(ind):
    w = rwide(ind); cols = [g for g in w.columns if g.startswith("IT")]
    first, lastyr = 2005 if 2005 in w.index else w.index.min(), w[cols].dropna(how="all").index.max()
    s = pd.DataFrame({"first": w.loc[first, cols], "last": w.loc[lastyr, cols]}); s["area"] = [area(g) for g in s.index]
    return s.groupby("area").median().loc[order].round(1), first, lastyr
for ind, t in [("nuts2_unemployment_rate", "Unemployment rate, %"), ("nuts2_employment_rate_20_64_female", "Female employment 20-64, %"), ("nuts2_tertiary_attainment_25_64", "Tertiary attainment 25-64, %")]:
    tab, a, b = italian(ind); print(f"{t} (median region, {a} -> {b})"); display(tab)'''),
    (C, '''fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
for ax, (ind, t) in zip(axes, [("nuts2_employment_rate_20_64_female", "Female employment 20-64, %"), ("nuts2_unemployment_rate", "Unemployment rate, %")]):
    w = rwide(ind)[[g for g in rwide(ind).columns if g.startswith("IT")]].dropna(how="all")
    palette = {"North-West": "#2a78d6", "North-East": "#2a78d6", "Centre": "#8a8a85", "South": "#eb6834", "Islands": "#eb6834"}
    for g in w.columns: ax.plot(w.index, w[g], color=palette[area(g)], lw=0.8, alpha=0.7)
    ax.set_title(t + " by region")
axes[0].plot([], [], color="#2a78d6", label="North"); axes[0].plot([], [], color="#8a8a85", label="Centre"); axes[0].plot([], [], color="#eb6834", label="South & Islands"); axes[0].legend(fontsize=8)
fig.tight_layout(); save(fig, "07_italy_regions_lines")'''),
]

NOTEBOOKS["07_regional_italy"].append((M, """## Takeaways (NUTS2, 2024)
- **A North-South cluster, not a capital-city outlier:** median regional GDP per head (PPS) is about 45-47k in the North, 38k in the Centre, 27k in the South and Islands (roughly 59% of the North). Regional dispersion measured by the coefficient of variation ranks only 11th of 20 countries (highest in Romania, Hungary, Slovakia), because those countries are driven by a rich capital region while Italy's gap is a broad regional split.
- **Labour gaps are larger than output gaps:** unemployment 8.2% (South) and 10.8% (Islands) vs 3.4-4.3% in the North (down from 13-14% in 2005); female employment 45-46% in the South vs 67-70% in the North.
- Tertiary attainment (25-64) is 22-25% in North and Centre vs 17.6-19.5% in the South and Islands: the educational gap is smaller than the employment gap."""))
