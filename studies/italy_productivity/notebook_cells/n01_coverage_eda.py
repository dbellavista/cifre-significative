"""Cells of notebook 01_coverage_eda."""
from . import NOTEBOOKS, C, LAST, M

NOTEBOOKS["01_coverage_eda"] = [
    (M, "# 01 - Data coverage\nWhat the panel contains, and where the gaps are, before any comparison."),
    (C, LAST + '''cov = (panel[panel.iso3.isin(EU27_ISO3 + BENCHMARKS) & panel.year.between(2005, LAST)]
       .groupby("indicator").agg(countries=("iso3", "nunique"), first_year=("year", "min"), last_year=("year", "max"),
                                    obs=("value", "size"), source=("source", "first")))
it = panel[(panel.iso3 == ITALY) & panel.year.between(2005, LAST)].groupby("indicator").year.agg(italy_years="nunique")
cov = cov.join(it)
display(cov.sort_values("italy_years").head(15))
print(len(cov), "indicators;", int((cov.italy_years < 20).sum()), "with fewer than 20 Italian years in 2005-2024")'''),
    (C, '''# share of the 20 years 2005-2024 available, per indicator and country group member (Italy + peers + benchmarks)
cols = [ITALY, *PEERS, *BENCHMARKS]
d = panel[panel.year.between(2005, LAST) & panel.iso3.isin(cols)]
share = (d.groupby(["indicator", "iso3"]).year.nunique().unstack() / 20).reindex(columns=cols).fillna(0)
fig, ax = plt.subplots(figsize=(8, 0.22 * len(share) + 1.2))
im = ax.imshow(share.values, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("s", ["#f0efec", "#2a78d6"]), vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(cols)), cols, rotation=90); ax.set_yticks(range(len(share)), share.index, fontsize=7); ax.grid(False)
ax.set_title("Share of 2005-2024 years with data (dark = complete)")
fig.colorbar(im, ax=ax, fraction=0.03); save(fig, "01_coverage")'''),
    (M, "**Reading the chart:** US/JP/GBR rows are empty for Eurostat-only series, so benchmark comparisons use the `imf_*`, `wb_*`, `ameco_*` indicators. Gaps for Italy itself are limited to `gini`, `arope` (2015+), PISA (triennial) and manually-entered series - see `docs/data_dictionary.md`."),
]
