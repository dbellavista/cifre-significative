# Cell definitions for scripts/make_notebooks.py: NOTEBOOKS[name] = [("md"|"code", text), ...]
NOTEBOOKS: dict[str, list[tuple[str, str]]] = {}
M, C = "md", "code"
LAST = "LAST = 2024  # last year with broad coverage (2025 is provisional for many series)\n"

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

NOTEBOOKS["02_italy_vs_eu"] = [
    (M, "# 02 - Italy vs the rest of the EU27: growth, productivity, convergence\nItaly in blue, the median of the other 26 members in orange (dashed), all other EU27 countries in gray."),
    (C, LAST + '''small_multiples(panel, [
    ("gdp_real_pc", "Real GDP per head", "EUR, chain-linked 2020"),
    ("gdp_growth", "Real GDP growth", "% per year"),
    ("employment_rate_20_64", "Employment rate 20-64", "% of population"),
    ("ameco_hours_per_worker", "Hours per worker", "hours per year"),
    ("ameco_tfp_index", "Total factor productivity", "2020 = 100"),
    ("nominal_ulc", "Nominal unit labour cost", "2020 = 100"),
], ncols=3, name="02_growth_overview")'''),
    (C, '''# GDP per head relative to Germany, France, Spain and the EU27 average (EU27 = 100, PPS)
rel = wide(panel, "gdp_pps_pc_eu27")
fig, ax = plt.subplots(figsize=(7.5, 4))
for c, lab in [("DEU", "Germany"), ("FRA", "France"), ("ESP", "Spain")]:
    ax.plot(rel.index, rel[c], color=PEER_C, lw=1.6); ax.annotate(lab, (rel.index[-1], rel[c].iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
ax.plot(rel.index, rel[ITALY], color=ITALY_C, lw=2.6); ax.annotate("Italy", (rel.index[-1], rel[ITALY].iloc[-1]), xytext=(4, 0), textcoords="offset points", color=ITALY_C, fontweight="bold", va="center")
ax.axhline(100, color=EU_C, lw=1.5, ls="--"); ax.annotate("EU27 = 100", (rel.index[0], 100), xytext=(2, 4), textcoords="offset points", color=EU_C, fontsize=8)
ax.set_title("GDP per head in PPS, EU27 = 100"); ax.set_ylabel("index"); save(fig, "02_convergence")
print(rel.loc[[2005, 2010, 2015, 2019, 2024], [ITALY, "DEU", "FRA", "ESP"]].round(1))'''),
    (M, "*Caveat:* the EU27 average itself rises as central/eastern members catch up, so part of Italy's fall in this index is their gain, not Italy's loss. Compare with Germany/France/Spain, which are the more telling references."),
    (C, '''# Where does the GDP-per-head gap come from? Log change 2005-2024 split into employment rate, hours, and a productivity residual
dec = gdp_pc_growth_decomposition(panel, 2005, 2024)
parts = ["employment rate", "hours per worker", "productivity & demography (residual)"]
show = dec.loc[["ITA", "DEU", "FRA", "ESP", "PRT", "GRC"]].copy()
show.loc["EU median (excl. IT)"] = dec.drop("ITA").median()
fig, ax = plt.subplots(figsize=(8, 4.2))
cols = ["#2a78d6", "#eda100", "#8a8a85"]
left_pos = np.zeros(len(show)); left_neg = np.zeros(len(show))
for part, col in zip(parts, cols):
    v = show[part].values; base = np.where(v >= 0, left_pos, left_neg)
    ax.barh(range(len(show)), v, left=base, color=col, height=0.62, label=part, edgecolor=SURFACE, linewidth=2)
    left_pos += np.where(v >= 0, v, 0); left_neg += np.where(v < 0, v, 0)
ax.scatter(show["real GDP per head"], range(len(show)), color=INK, zorder=5, s=28, label="real GDP per head (total)")
ax.set_yticks(range(len(show)), show.index); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("log points x100, 2005-2024"); ax.set_title("Where growth in GDP per head came from, 2005-2024")
ax.legend(loc="lower right", fontsize=8); save(fig, "02_growth_decomposition")
display(show.round(1))'''),
    (C, '''# Italy's rank among EU27 members, selected years (1 = highest; index series such as TFP and ULC are 2020=100, so ranks are not meaningful)
rows = {}
for ind in ["gdp_real_pc", "gdp_growth", "employment_rate_20_64", "employment_rate_20_64_female", "ameco_hours_per_worker", "hicp_inflation"]:
    w = wide(panel, ind)
    rows[ind] = {y: "%d/%d" % italy_rank(w, y) for y in (2005, 2010, 2015, 2019, 2024) if not np.isnan(w.loc[y, ITALY])}
display(pd.DataFrame(rows).T)'''),
]

NOTEBOOKS["03_debt_investment"] = [
    (M, "# 03 - Debt, interest and investment\nIs Italy's debt a story of deficits, or of interest and growth?"),
    (C, '''small_multiples(panel, [
    ("gov_debt_pct_gdp", "Gross government debt", "% of GDP"),
    ("gov_interest_pct_gdp", "Interest paid by government", "% of GDP"),
    ("long_term_yield", "10-year government bond yield", "%"),
    ("gfcf_pct_gdp", "Gross fixed capital formation", "% of GDP"),
    ("gov_pubinv_pct_gdp", "Public investment", "% of GDP"),
    ("rd_pct_gdp", "R&D expenditure", "% of GDP"),
], ncols=3, name="03_debt_investment_overview")'''),
    (C, '''# Primary balance (balance + interest) and the interest-growth differential
bal = wide(panel, "gov_balance_pct_gdp"); intr = wide(panel, "gov_interest_pct_gdp")
pb = (bal + intr).loc[2005:2024]
print("Average primary balance 2005-2024 (% GDP):", pb.mean()[[ITALY, "DEU", "FRA", "ESP", "PRT", "GRC"]].round(2).to_dict(),
      "| median of other EU27:", round(pb.mean().drop(["ITA", "GBR", "USA", "JPN", "EU27"], errors="ignore").median(), 2))
rg = (implicit_interest_rate(panel) - nominal_growth(panel)).loc[2006:2024]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
w = pb; med = eu_median_ex_italy(w)
for c in EU27_ISO3:
    if c != ITALY: axes[0].plot(w.index, w[c], color=PEER_C, lw=0.7, alpha=0.7)
axes[0].plot(med.index, med, color=EU_C, lw=2, ls="--"); axes[0].plot(w.index, w[ITALY], color=ITALY_C, lw=2.4); axes[0].axhline(0, color=INK2, lw=0.8)
axes[0].set_title("Primary balance"); axes[0].set_ylabel("% of GDP"); axes[0].set_ylim(-12, 8)
rgm = rg.rolling(3, center=True, min_periods=1).mean()
for c in EU27_ISO3:
    if c != ITALY: axes[1].plot(rgm.index, rgm[c], color=PEER_C, lw=0.7, alpha=0.7)
axes[1].plot(rgm.index, rgm.drop(columns=[ITALY]).median(axis=1), color=EU_C, lw=2, ls="--"); axes[1].plot(rgm.index, rgm[ITALY], color=ITALY_C, lw=2.4); axes[1].axhline(0, color=INK2, lw=0.8)
axes[1].set_title("Implicit interest rate minus nominal growth (3-yr avg)"); axes[1].set_ylabel("pp"); axes[1].set_ylim(-8, 8)
fig.suptitle("Italy (blue), median of others (orange dashed), other EU27 (gray)", fontsize=9, color=INK2, x=0.01, ha="left"); fig.tight_layout(); save(fig, "03_primary_balance_rg")'''),
    (C, '''# What moved debt/GDP? Contributions in pp of GDP, two sub-periods
countries = ["ITA", "DEU", "FRA", "ESP", "PRT", "GRC"]
periods = [(2005, 2014), (2014, 2024)]
parts = ["primary balance", "interest", "nominal growth", "other (stock-flow)"]
cols = ["#2a78d6", "#e34948", "#1baf7a", "#b9b8b2"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
for ax, (a, b) in zip(axes, periods):
    dec = pd.DataFrame({c: debt_decomposition(panel, c, a, b) for c in countries}).T
    pos = np.zeros(len(dec)); neg = np.zeros(len(dec))
    for part, col in zip(parts, cols):
        v = dec[part].values; base = np.where(v >= 0, pos, neg)
        ax.bar(range(len(dec)), v, bottom=base, color=col, label=part, edgecolor=SURFACE, linewidth=2, width=0.65)
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.scatter(range(len(dec)), dec.sum(axis=1), color=INK, zorder=5, s=26, label="change in debt/GDP")
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(range(len(dec)), dec.index); ax.set_title(f"{a}-{b}: change in debt/GDP, pp"); ax.margins(y=0.1)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.04)); fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, "03_debt_decomposition")
display(pd.DataFrame({f"{a}-{b}": {c: round(debt_decomposition(panel, c, a, b).sum(), 1) for c in countries} for a, b in periods}))'''),
    (C, '''# Spread of Italian vs German 10-year yields, and debt of the US and Japan for scale (IMF, general government)
y = wide(panel, "long_term_yield")
spread = (y[ITALY] - y["DEU"]).dropna()
imfd = wide(panel, "imf_gov_debt_pct_gdp", [ITALY, "DEU", "FRA", "ESP", "USA", "JPN", "GBR"])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].fill_between(spread.index, spread, color=ITALY_C, alpha=0.25); axes[0].plot(spread.index, spread, color=ITALY_C, lw=2); axes[0].set_title("Italy-Germany 10-year yield spread"); axes[0].set_ylabel("pp")
for c, col, ls in [("USA", PEER_C, "-"), ("JPN", PEER_C, "-"), ("GBR", PEER_C, "-"), ("DEU", PEER_C, "-"), ("FRA", PEER_C, "-")]:
    axes[1].plot(imfd.index, imfd[c], color=col, lw=1.3); axes[1].annotate(c, (imfd.index[-1], imfd[c].iloc[-1]), xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK2, va="center")
axes[1].plot(imfd.index, imfd[ITALY], color=ITALY_C, lw=2.5); axes[1].annotate("ITA", (imfd.index[-1], imfd[ITALY].iloc[-1]), xytext=(3, 0), textcoords="offset points", color=ITALY_C, fontweight="bold", va="center")
axes[1].set_title("General government gross debt (IMF)"); axes[1].set_ylabel("% of GDP"); fig.tight_layout(); save(fig, "03_spread_benchmarks")
print("Peak spread:", spread.idxmax(), round(spread.max(), 2), "| 2024:", round(spread.loc[2024], 2))'''),
]

NOTEBOOKS["04_demography_pensions"] = [
    (M, "# 04 - Demography and pensions"),
    (C, '''small_multiples(panel, [
    ("old_age_dependency", "Old-age dependency ratio (65+ per 100 aged 15-64)", "ratio"),
    ("median_age", "Median age", "years"),
    ("fertility_rate", "Total fertility rate", "births per woman"),
    ("pension_spend_pct_gdp", "Public spending on old age (COFOG)", "% of GDP"),
    ("employment_rate_55_64", "Employment rate 55-64", "%"),
    ("oecd_exit_age_male", "Effective labour-market exit age, men", "years"),
], ncols=3, name="04_demography_pensions_overview")'''),
    (C, '''# Does Italy spend more on pensions than its age structure explains? (EU27, 2022-2024 average)
win = slice(2022, 2024)
x = wide(panel, "old_age_dependency", EU27_ISO3).loc[win].mean(); y = wide(panel, "pension_spend_pct_gdp", EU27_ISO3).loc[win].mean()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
scatter_italy(axes[0], x, y, "Old-age dependency ratio", "Old-age public spending, % of GDP", "Pension spending vs ageing, 2022-24")
e = wide(panel, "employment_rate_55_64", EU27_ISO3).loc[win].mean()
scatter_italy(axes[1], e, y, "Employment rate 55-64, %", "Old-age public spending, % of GDP", "Pension spending vs older-worker employment")
fig.tight_layout(); save(fig, "04_pension_scatter")
b = np.polyfit(x.drop(ITALY).dropna(), y.drop(ITALY).dropna().reindex(x.drop(ITALY).dropna().index), 1)
print("Italy spends %.1f%% of GDP; peers' ageing-line predicts %.1f%%" % (y[ITALY], np.polyval(b, x[ITALY])))'''),
    (C, '''# Pension design in 2024 (OECD Pensions at a Glance): replacement rates and retirement ages
s = pd.read_parquet(PROCESSED / "pensions_oecd_2024.parquet")
sel = s[s.REF_AREA.isin(EU27_ISO3 + BENCHMARKS) & s.MEASURE.isin(["GPRR100", "NPRR100", "FRPLF22", "CRPLF22"]) & (s.SEX.isin(["M", "_Z"]))]
tab = sel.drop_duplicates(["REF_AREA", "MEASURE"]).pivot(index="REF_AREA", columns="MEASURE", values="OBS_VALUE")
tab = tab.rename(columns={"GPRR100": "gross replacement rate, avg earner (%)", "NPRR100": "net replacement rate, avg earner (%)",
                          "CRPLF22": "normal retirement age, current (men)", "FRPLF22": "normal retirement age, future (men)"})
rank = tab.rank(ascending=False)
print("Italy rank among", len(tab), "countries (1 = highest):"); display(pd.DataFrame({"value": tab.loc[ITALY], "rank": rank.loc[ITALY]}))
display(tab.describe().loc[["min", "50%", "max"]].round(1))'''),
    (C, '''# Old-age poverty (at-risk-of-poverty rate, 65+, 60% of median)
small_multiples(panel, [("arop_65plus", "At-risk-of-poverty rate, 65+", "% of 65+")], ncols=1, name="04_arop65")'''),
]

NOTEBOOKS["05_education_labour"] = [
    (M, "# 05 - Education, labour market and wages"),
    (C, '''small_multiples(panel, [
    ("tertiary_attainment_25_34", "Tertiary attainment, 25-34", "% of age group"),
    ("early_leavers_18_24", "Early school leavers, 18-24", "%"),
    ("neet_rate_15_29", "NEET rate, 15-29", "%"),
    ("edu_spend_pct_gdp", "Public spending on education", "% of GDP"),
    ("youth_unemployment_rate", "Youth unemployment (15-24)", "%"),
    ("employment_rate_20_64_female", "Female employment rate 20-64", "%"),
], ncols=3, name="05_education_labour_overview")'''),
    (C, '''# PISA (World Bank EdStats up to 2018; 2022 from the OECD workbook via data/external/pisa_2022.csv)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
for ax, (ind, t) in zip(axes, [("wb_pisa_math", "PISA mathematics"), ("wb_pisa_reading", "PISA reading"), ("wb_pisa_science", "PISA science")]):
    w = wide(panel, ind).dropna(how="all")
    for c in EU27_ISO3:
        if c != ITALY and c in w: ax.plot(w.index, w[c], color=PEER_C, lw=0.7, marker="o", ms=2.5)
    ax.plot(w.index, eu_median_ex_italy(w), color=EU_C, lw=2, ls="--", marker="o", ms=3); ax.plot(w.index, w[ITALY], color=ITALY_C, lw=2.4, marker="o", ms=4); ax.set_title(t); ax.set_xticks(w.index)
axes[0].set_ylabel("mean score"); fig.tight_layout(); save(fig, "05_pisa")'''),
    (C, '''# Real compensation per employee (AMECO nominal compensation, deflated with cumulative HICP), 2005 = 100. Approximate: total economy, composition effects included.
comp = wide(panel, "ameco_comp_per_employee_ncu"); infl = wide(panel, "hicp_inflation")
price = (1 + infl / 100).cumprod(); price = price / price.loc[2005]
real = (comp / comp.loc[2005]) / price * 100
real = real.loc[2005:2024]
fig, ax = plt.subplots(figsize=(7.5, 4))
for c in EU27_ISO3:
    if c != ITALY and c in real: ax.plot(real.index, real[c], color=PEER_C, lw=0.7, alpha=0.7)
for c, lab in [("DEU", "Germany"), ("FRA", "France"), ("ESP", "Spain")]:
    ax.annotate(lab, (2024, real.loc[2024, c]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
ax.plot(real.index, eu_median_ex_italy(real), color=EU_C, lw=2, ls="--"); ax.plot(real.index, real[ITALY], color=ITALY_C, lw=2.5)
ax.annotate("Italy", (2024, real.loc[2024, ITALY]), xytext=(4, 0), textcoords="offset points", color=ITALY_C, fontweight="bold", va="center")
ax.set_ylim(60, 260); ax.set_title("Real compensation per employee, 2005 = 100"); ax.set_ylabel("index"); save(fig, "05_real_compensation")
print(real.loc[2024, [ITALY, "DEU", "FRA", "ESP"]].round(1).to_dict(), "EU median ex IT:", round(eu_median_ex_italy(real).loc[2024], 1))'''),
    (C, '''# Human capital vs income: tertiary attainment 25-34 and GDP per head (EU27, 2022-24 mean)
win = slice(2022, 2024)
x = wide(panel, "tertiary_attainment_25_34", EU27_ISO3).loc[win].mean(); y = wide(panel, "gdp_real_pc", EU27_ISO3).loc[win].mean() / 1000
fig, ax = plt.subplots(figsize=(6.5, 4.4)); scatter_italy(ax, x, y, "Tertiary attainment 25-34, %", "Real GDP per head, EUR thousand", "Education level vs income, 2022-24"); save(fig, "05_tertiary_gdp")
print("Correlation across EU27:", round(x.corr(y), 2))'''),
]

NOTEBOOKS["06_correlations"] = [
    (M, "# 06 - Correlations, panel regression, and which countries look like Italy\nAssociations, not causal claims."),
    (C, '''# 1) Where does Italy sit? Gap vs the median of other EU27, in standard deviations, three periods
groups = {g: it for g, it in GAP_GROUPS.items() if g not in ("Gender & wages", "Governance")}  # shared definition (cifresignificative.analysis.data); the two newer groups are shown in notebooks 09 and 11
windows = {"2005-09": (2005, 2009), "2015-19": (2015, 2019), "2020-24": (2020, 2024)}
rows = {}
for grp, items in groups.items():
    for ind, label in items:
        rows[(grp, label)] = {k: zscore_gap(panel, [ind], w).get(ind, np.nan) for k, w in windows.items()}
gaps = pd.DataFrame(rows).T
gaps.index = [f"{l}" for g, l in gaps.index]
gap_heatmap(gaps, "Italy vs the other EU27 members", name="06_gap_heatmap")
display(gaps.round(1))'''),
    (C, '''# 2) Cross-country correlations of changes 2005-2019 -> are faster-growing countries different?
inds = ["gdp_real_pc", "gfcf_pct_gdp", "gov_debt_pct_gdp", "employment_rate_20_64", "tertiary_attainment_25_34", "old_age_dependency", "rd_pct_gdp", "early_leavers_18_24", "edu_spend_pct_gdp"]
ch = {}
for ind in inds:
    w = wide(panel, ind, EU27_ISO3)
    ch[ind] = (np.log(w.loc[2019] / w.loc[2005]) * 100) if ind == "gdp_real_pc" else (w.loc[2019] - w.loc[2005])
ch = pd.DataFrame(ch).dropna()
corr = ch.corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks(range(len(inds)), inds, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(len(inds)), inds, fontsize=8)
for i in range(len(inds)):
    for j in range(len(inds)): ax.text(j, i, f"{corr.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
ax.set_title("Correlation of changes 2005-2019 across EU27 (n=%d)" % len(ch)); fig.colorbar(im, ax=ax, fraction=0.04); save(fig, "06_change_correlations")'''),
    (C, '''# 3) Panel regression with country and year fixed effects: annual real growth on lagged levels (EU27, 2006-2024)
from linearmodels.panel import PanelOLS
vars_ = {"gov_debt_pct_gdp": "debt/GDP (lag)", "gfcf_pct_gdp": "investment/GDP (lag)", "employment_rate_20_64": "employment rate (lag)",
         "tertiary_attainment_25_34": "tertiary 25-34 (lag)", "old_age_dependency": "old-age dependency (lag)"}
d = panel[panel.iso3.isin(EU27_ISO3) & panel.year.between(2005, 2024)]
wide_all = d.pivot_table(index=["iso3", "year"], columns="indicator", values="value")
X = wide_all[list(vars_)].groupby(level=0).shift(1).rename(columns=vars_)
df = pd.concat([wide_all["gdp_growth"].rename("growth"), X], axis=1).dropna()
out = {}
for lab, sub in [("all EU27", df), ("excluding Ireland", df[~df.index.get_level_values("iso3").isin(["IRL"])])]:
    r = PanelOLS(sub["growth"], sub.drop(columns="growth"), entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    out[lab] = pd.DataFrame({"coef": r.params, "p": r.pvalues}).round(3).assign(n=int(r.nobs), within_R2=round(r.rsquared_within, 3))
tab = pd.concat(out, axis=1); display(tab)
res = r  # last fit (excluding Ireland), used in the line below
print("n (excluding Ireland) =", int(res.nobs), "| within R2 =", round(res.rsquared_within, 3))'''),
    (C, '''# 4) Which countries look most like Italy? Euclidean distance on standardized 2022-24 profile (EU27)
prof = ["gdp_real_pc", "gov_debt_pct_gdp", "gfcf_pct_gdp", "employment_rate_20_64", "employment_rate_55_64", "old_age_dependency", "pension_spend_pct_gdp", "tertiary_attainment_25_34", "early_leavers_18_24", "neet_rate_15_29", "edu_spend_pct_gdp", "rd_pct_gdp"]
P = pd.DataFrame({i: wide(panel, i, EU27_ISO3).loc[2022:2024].mean() for i in prof}).dropna()
Z = (P - P.mean()) / P.std()
dist = ((Z - Z.loc[ITALY]) ** 2).sum(axis=1).pow(0.5).drop(ITALY).sort_values()
from sklearn.decomposition import PCA
pcs = pd.DataFrame(PCA(2).fit_transform(Z), index=Z.index, columns=["PC1", "PC2"])
fig, ax = plt.subplots(figsize=(6.2, 4.6)); ax.scatter(pcs.PC1, pcs.PC2, color=PEER_C, s=34, edgecolor=SURFACE)
for c, r in pcs.iterrows(): ax.annotate(c, (r.PC1, r.PC2), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
ax.scatter(*pcs.loc[ITALY], color=ITALY_C, s=90, zorder=4, edgecolor=SURFACE); ax.annotate("Italy", pcs.loc[ITALY], xytext=(6, 5), textcoords="offset points", color=ITALY_C, fontweight="bold")
ax.set_title("EU27 country profiles, first two principal components"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); save(fig, "06_pca_profiles")
print("Closest profiles to Italy:", dist.head(5).round(2).to_dict(), "\\nFurthest:", dist.tail(3).round(2).to_dict())'''),
]

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

NOTEBOOKS["02_italy_vs_eu"].append((M, """## Takeaways (numbers from the cells above; Eurostat/AMECO, 2005-2024)
- **Italy is not poor, it is stagnant.** Real GDP per head is mid-table (11th-12th of 27 throughout) but grew only +5% (log points) over 2005-2024, against +27% for the median of the other members, +20% Germany, +17% Portugal, +12.5% France, +11% Spain. Of the 27 members with data only Greece (-3%) and Luxembourg (+4.4%, from a very high base) were lower.
- **The shortfall is in the productivity/demography residual**, not hours or employment: +0.2 for Italy vs +18.4 for the median. Employment rose +8.7 (median +9.6) and hours fell -3.7 (median -4.5). The residual mixes output per hour with age-structure effects, so it cannot say how much is each.
- **Employment is the structural gap in levels:** 20-64 employment was 67.1% in 2024 against a 78.4% median (27th of 27); female employment 57.4% vs 75.2% (also last).
- **Convergence went backwards:** GDP per head in PPS (EU27 = 100) fell from 116.6 to 93.5, while Germany stayed near 130. Part of this is eastern members catching up, but Spain and France show smaller falls than Italy relative to Germany."""))

NOTEBOOKS["03_debt_investment"].append((M, """## Takeaways (Eurostat general-government data; IMF for US/JP)
- Debt was 106% of GDP in 2005 and 135% in 2024 (155% at the 2020 peak): the second highest in the EU27 in 2024, with interest at 3.9% of GDP against a 1.3% median.
- **It is not a primary-deficit story.** Italy's average primary balance over 2005-2024 was +0.1% of GDP, versus -0.9% for the median, +0.6% Germany, -2.6% France, -2.7% Spain.
- **It is an interest-and-growth story.** In 2005-2014 debt rose 28.6 pp: interest added +41 pp, offset by primary surpluses (-13) and nominal growth (-9). In 2014-2024 debt was flat (-0.1 pp): interest still added +36 pp, but nominal growth (-42) paid it back, and primary balances turned to deficits (+11 pp).
- The Italy-Germany 10-year spread peaked at 4.0 pp in 2012 and was 1.4 pp in 2024.
- **Public investment** (3.6% of GDP in 2024 vs 4.2% median) was much lower in 2012-2019 (2.1-2.6% of GDP vs a 3.2-3.9% median); total investment is back in line with the median (21.6% vs 21.2%), having dipped to about 17% in 2014.
- The "other" bar is the stock-flow residual (debt changes not explained by deficit, interest and growth)."""))

NOTEBOOKS["04_demography_pensions"].append((M, """## Takeaways
- **Italy is the oldest EU27 country:** old-age dependency 38.4 (1st of 27), median age 48.7 (1st), fertility 1.2 (22nd).
- **Spending is above what ageing alone predicts.** Public old-age spending was 13.7% of GDP in 2022-24, 3 pp above the 10.8% implied by the other members' relationship with the dependency ratio (3rd highest in 2024). The association with older-worker employment is negative: Italy combines high spending with low 55-64 employment (59.0% vs 67.8% median in 2024).
- **But Italy has moved a lot:** 55-64 employment rose from 31.4% (2005) to 59.0%, and the effective exit age for men from 59.9 to 64.0 (EU median 63.6).
- **Design in 2024 (OECD):** net replacement rate for an average earner 79% (8th of 30; median 72%), gross 70.6% (8th; median 54%); normal retirement age 64.8 now but 70 for future entrants (rank 4-5 of 30). The retirement-age path is the main reform lever already legislated.
- **Old-age poverty is not unusual:** at-risk-of-poverty among 65+ was 17.6% against a 16.6% median in 2024 (gap about 0 sd). The pension system protects the old comparatively well; the cost falls on public finances and on younger cohorts (see notebook 05)."""))

NOTEBOOKS["05_education_labour"].append((M, """## Takeaways
- **Tertiary attainment (25-34):** up from 16.2% to 31.6% but 26th of 27 (median 44.8%). Cross-country correlation with GDP per head: 0.62. Public education spending is 4.0% of GDP vs a 5.0% median (24th).
- **Early leavers fell sharply** (22.1% -> 9.8%, median 7.9%): this gap has nearly closed.
- **NEET (15-29)** was 25.7% in 2015 and is 15.2% in 2024, still the 2nd highest in the EU27 (median 10.0%). Youth unemployment is 20.3% vs 15.7%.
- **PISA:** Italian scores rose between 2006 and 2018 (maths 462 -> 487, reading 469 -> 476) and fell in 2022 to 471 in maths, 482 in reading (477 in science), a post-pandemic dip seen across countries. Germany's maths score fell from 504 (2006) to 475, so the gap with Italy went from 42 points to 4; in reading Italy (482) is now slightly above Germany (480) and France (474). Italy ranks 19th of 26 EU members in maths and 8th in reading in 2022. Spain's 2018 reading score is missing in the World Bank series. 2022 comes from the OECD workbook (docs/manual_data.md).
- **Real compensation per employee** (AMECO nominal compensation deflated with HICP, 2005 = 100) was 90.5 in 2024 against 110.0 Germany, 103.8 France, 105.4 Spain and a 118 median: Italy is the weak end, but note this is a total-economy average with composition effects, and the 2022-23 inflation spike drives much of the last years."""))

NOTEBOOKS["06_correlations"].append((M, """## Takeaways
- **Italy's profile against the other EU27 (heatmap, 2020-24):** furthest below the median on employment (-2.8 sd) and female employment (-3.1 sd); furthest above on interest costs (+3.2), debt (+2.4), NEET (+2.8), old-age spending (+1.8) and old-age dependency (+1.4); also below on tertiary attainment (-1.5) and education spending (-1.0). GDP per head is slightly *above* the median (+0.3) and R&D, 10-year yields, and 65+ poverty are close to it.
- **Growth regression is weak and fragile:** with country and year fixed effects, lagged investment/GDP appears significant (+0.21 pp growth per pp, p=0.02) only when Ireland is included; excluding Ireland (whose GDP is dominated by multinational accounting) it is +0.08 (p=0.19), and no regressor is significant in either 3-year or 5-year windows (notebook 20). Treat the regression as showing no robust association rather than a finding about investment.
- **Peer countries by profile:** Greece, France, Spain, Portugal and Romania are closest to Italy; Luxembourg, Sweden and Ireland are furthest."""))

NOTEBOOKS["07_regional_italy"].append((M, """## Takeaways (NUTS2, 2024)
- **A North-South cluster, not a capital-city outlier:** median regional GDP per head (PPS) is about 45-47k in the North, 38k in the Centre, 27k in the South and Islands (roughly 59% of the North). Regional dispersion measured by the coefficient of variation ranks only 11th of 20 countries (highest in Romania, Hungary, Slovakia), because those countries are driven by a rich capital region while Italy's gap is a broad regional split.
- **Labour gaps are larger than output gaps:** unemployment 8.2% (South) and 10.8% (Islands) vs 3.4-4.3% in the North (down from 13-14% in 2005); female employment 45-46% in the South vs 67-70% in the North.
- Tertiary attainment (25-64) is 22-25% in North and Centre vs 17.6-19.5% in the South and Islands: the educational gap is smaller than the employment gap."""))

NOTEBOOKS["08_productivity_sectors"] = [
    (M, "# 08 - Productivity by sector\nReal gross value added per hour worked (Eurostat national accounts, chain-linked 2020 euro), market sectors only (real estate excluded because of imputed rents). Only growth is compared across countries: euro levels would need PPPs."),
    (C, '''sectors = load_sectors()
agg = aggregate_productivity(sectors)
cover = agg[[c for c in agg if c in EU27_ISO3]].notna().sum(axis=1)
agg = agg.loc[:cover[cover >= 20].index.max()]  # stop at the last year with broad country coverage
idx = agg / agg.loc[2005] * 100
fig, ax = plt.subplots(figsize=(7.5, 4))
for c in EU27_ISO3:
    if c != ITALY and c in idx: ax.plot(idx.index, idx[c], color=PEER_C, lw=0.7, alpha=0.7)
for c, lab in [("DEU", "Germany"), ("FRA", "France"), ("ESP", "Spain")]:
    ax.annotate(lab, (idx[c].dropna().index.max(), idx[c].dropna().iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
med = idx[[c for c in idx if c in EU27_ISO3 and c != ITALY]].median(axis=1)
ax.plot(med.index, med, color=EU_C, lw=2, ls="--"); ax.plot(idx.index, idx[ITALY], color=ITALY_C, lw=2.5)
ax.annotate("Italy", (idx[ITALY].dropna().index.max(), idx[ITALY].dropna().iloc[-1]), xytext=(4, 0), textcoords="offset points", color=ITALY_C, fontweight="bold", va="center")
lo, hi = np.nanpercentile(idx.values, [1, 99]); ax.set_ylim(lo - 5, hi + 5)
ax.set_title("Real output per hour, market sectors (2005 = 100)"); ax.set_ylabel("index"); save(fig, "08_productivity_index")
y1 = int(agg[ITALY].dropna().index.max()); print("Last year:", y1, "| Italy:", round(idx.loc[y1, ITALY], 1), "| median of others:", round(med.loc[y1], 1), "| DEU/FRA/ESP:", idx.loc[y1, ["DEU", "FRA", "ESP"]].round(1).to_dict())'''),
    (C, '''# Sector productivity growth, 2005-2025 (average annual log growth), Italy vs the median of the other members; same sample as the shift-share below (members with all nine sections in both years)
A, B = 2005, 2025  # 2025 is a first estimate (Eurostat release of 21 Oct 2026): Bulgaria and Latvia have no 2025 value added
SAMPLE = [c for c in EU27_ISO3 if c in sectors.iso3.unique() and shift_share(sectors, c, A, B).notna().all()]; print("members in the sample:", len(SAMPLE), "| left out:", sorted(set(EU27_ISO3) - set(SAMPLE)))
def sector_growth(a, b):
    p = sectors.pivot_table(index="iso3", columns=["nace_r2", "year"], values="prod")
    return pd.DataFrame({n: np.log(p[(n, b)] / p[(n, a)]) / (b - a) * 100 for n in MARKET_SECTORS})
g = sector_growth(A, B).loc[SAMPLE]
tab = pd.DataFrame({"Italy": g.loc[ITALY], "median of others": g.drop(ITALY).median(), "Germany": g.loc["DEU"]}).rename(index=SECTOR_LABELS).sort_values("Italy")
fig, ax = plt.subplots(figsize=(8, 4.2)); yy = np.arange(len(tab))
ax.scatter(tab["median of others"], yy, color=EU_C, s=40, zorder=3, label="median of other EU27", marker="D")
ax.scatter(tab["Germany"], yy, color=PEER_C, s=36, zorder=2, label="Germany")
ax.barh(yy, tab["Italy"], color=ITALY_C, height=0.45, zorder=2, label="Italy")
ax.set_yticks(yy, tab.index); ax.axvline(0, color=INK2, lw=0.8); ax.set_xlabel("% per year, real value added per hour, 2005-2025")
ax.set_title("Sector productivity growth, 2005-2025"); ax.legend(fontsize=8, loc="lower right"); save(fig, "08_sector_growth"); display(tab.round(2))'''),
    (C, '''# Shift-share: within-sector productivity vs reallocation of hours across sectors (percent change in aggregate productivity)
countries = ["ITA", "DEU", "FRA", "ESP", "PRT", "GRC"]
rows = {c: shift_share(sectors, c, A, B) for c in countries}
oth = pd.DataFrame({c: shift_share(sectors, c, A, B) for c in SAMPLE if c != ITALY}).T; print('median total change (the parts do not add up to it):', round(oth.sum(axis=1).median(), 1), '| countries:', len(oth)); oth = oth.median()
dec = pd.DataFrame(rows).T; dec.loc["EU median (excl. IT)"] = oth
fig, ax = plt.subplots(figsize=(8, 3.8)); cols = ["#2a78d6", "#eda100", "#b9b8b2"]
pos = np.zeros(len(dec)); neg = np.zeros(len(dec))
for part, col in zip(dec.columns, cols):
    v = dec[part].values; base = np.where(v >= 0, pos, neg)
    ax.barh(range(len(dec)), v, left=base, color=col, height=0.6, label=part, edgecolor=SURFACE, linewidth=2)
    pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
ax.scatter(dec.sum(axis=1), range(len(dec)), color=INK, s=26, zorder=5, label="total")
ax.set_yticks(range(len(dec)), dec.index); ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("% change in aggregate output per hour, 2005-2025"); ax.set_title("Where productivity growth came from, 2005-2025"); ax.legend(fontsize=8, loc="lower right"); save(fig, "08_shift_share")
display(dec.round(1))'''),
    (C, '''# Structure: share of hours by sector, 2005 vs latest year with data (Italy vs median of the others)
def hshare(y):
    h = sectors.pivot_table(index="iso3", columns=["nace_r2", "year"], values="hours_ths")
    t = pd.DataFrame({n: h[(n, y)] for n in MARKET_SECTORS}); return t.div(t.sum(axis=1), axis=0) * 100
a, b = hshare(A), hshare(B)
out = pd.DataFrame({"Italy 2005": a.loc[ITALY], "Italy 2025": b.loc[ITALY], "others 2005 (median)": a.drop(ITALY).median(), "others 2025 (median)": b.drop(ITALY).median()}).rename(index=SECTOR_LABELS)
display(out.round(1))'''),
]

NOTEBOOKS["09_governance_evasion"] = [
    (M, "# 09 - Governance, tax compliance and public procurement\nProxies for administrative quality and for 'bad spending'. Governance scores are perception-based; the VAT gap is an estimate; procurement competition comes from the Single Market Scoreboard (2021-2024)."),
    (C, '''small_multiples(panel, [
    ("wgi_government_effectiveness", "Government effectiveness", "score 0-100"),
    ("wgi_regulatory_quality", "Regulatory quality", "score 0-100"),
    ("wgi_rule_of_law", "Rule of law", "score 0-100"),
    ("wgi_control_of_corruption", "Control of corruption", "score 0-100"),
    ("vat_compliance_gap_pct", "VAT compliance gap", "% of VAT due"),
], ncols=3, name="09_governance_overview")
w = wide(panel, "vat_compliance_gap_pct"); print("VAT gap Italy:", w[ITALY].loc[[2013, 2019, 2023]].round(1).to_dict(), "| median of others:", eu_median_ex_italy(w).loc[[2013, 2019, 2023]].round(1).to_dict(), "| rank 2023:", italy_rank(w, 2023))'''),
    (C, '''# Do cleaner/more effective governments collect more of the VAT due? (2023; EU27, Luxembourg excluded as unreliable in the source)
x = wide(panel, "wgi_control_of_corruption", EU27_ISO3).loc[2023]; y = wide(panel, "vat_compliance_gap_pct", EU27_ISO3).loc[2023]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
scatter_italy(axes[0], x, y, "Control of corruption (0-100)", "VAT compliance gap, % of VAT due", "VAT gap vs control of corruption, 2023")
ge = wide(panel, "wgi_government_effectiveness", EU27_ISO3).loc[2023]; inc = wide(panel, "gdp_pps_pc_eu27", EU27_ISO3).loc[2023]
scatter_italy(axes[1], ge, inc, "Government effectiveness, 2023 (0-100)", "GDP per head in PPS, EU27 = 100", "Government effectiveness vs income level, 2023")
fig.tight_layout(); save(fig, "09_governance_scatter")
print("corr(VAT gap, corruption control):", round(x.corr(y), 2), "| corr(gov. effectiveness, income level):", round(ge.corr(inc), 2))'''),
    (C, '''# Public procurement competition (Single Market Scoreboard, from TED; 2021-2024)
items = [
    (wide(panel, "proc_single_bidder_pct"), "Contracts with a single bidder", "% of contracts"),
    (wide(panel, "proc_direct_awards_pct"), "Direct awards (no call for bids)", "% of procedures"),
    (wide(panel, "proc_decision_speed_days"), "Decision speed", "days"),
    (wide(panel, "proc_price_only_award_pct"), "Awards on price alone", "% of contracts"),
]
small_multiples_wide(items, ncols=4, name="09_procurement")
for ind in ["proc_single_bidder_pct", "proc_direct_awards_pct", "proc_decision_speed_days"]:
    w = wide(panel, ind)
    print(ind, "| Italy 2024:", round(w.loc[2024, ITALY], 1), "| median of others:", round(eu_median_ex_italy(w).loc[2024], 1), "| rank (1 = highest): %d/%d" % italy_rank(w, 2024))'''),
    (C, '''# Italy only: the Ministry of Economy's tax-gap estimates (2018-2022) and Istat's irregular-work rate
mef = pd.read_parquet(PROCESSED / "italy_tax_gap_mef.parquet"); irr = pd.read_parquet(PROCESSED / "istat_irregular_work.parquet")
gap = mef[(mef.measure == "gap_meur") & mef.tax.isin(["IRPEF lavoro autonomo e impresa", "IVA", "IRES", "IRAP", "IMU-TASI"])].pivot(index="year", columns="tax", values="value") / 1000
prop = mef[(mef.measure == "propensity_pct") & mef.tax.isin(["Totale entrate tributarie", "IVA", "IRPEF lavoro autonomo e impresa"])].pivot(index="year", columns="tax", values="value")
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
bottom = np.zeros(len(gap))
for col, colr in zip(["IRPEF lavoro autonomo e impresa", "IVA", "IRES", "IRAP", "IMU-TASI"], [ITALY_C, EU_C, "#1baf7a", "#eda100", PEER_C], strict=True):
    axes[0].bar(gap.index, gap[col], bottom=bottom, color=colr, label=col.replace("lavoro autonomo e impresa", "self-employed & business"), width=0.65, edgecolor=SURFACE); bottom += gap[col].values
axes[0].set_title("Tax gap by tax, EUR billion"); axes[0].legend(fontsize=7, loc="lower left")
for col, colr in zip(prop.columns, [INK2, EU_C, ITALY_C], strict=True): axes[1].plot(prop.index, prop[col], marker="o", color=colr, label=col.replace("lavoro autonomo e impresa", "self-employed & business"))
axes[1].set_title("Propensity to evade, % of tax due"); axes[1].legend(fontsize=7)
tot = irr[(irr.industry == "_T") & (irr.measure == "RUEP")].set_index("year").value; axes[2].plot(tot.index, tot.values, color=ITALY_C, lw=2.4); axes[2].set_title("Irregular employment rate (Istat), %"); axes[2].set_ylim(8, 13)
for ax in axes: ax.set_xticks(sorted(set(mef.year)) if ax is not axes[2] else [2005, 2010, 2015, 2020, 2023])
fig.tight_layout(); save(fig, "09_italy_tax_gap")
display(mef[mef.measure == "gap_meur"].pivot(index="tax", columns="year", values="value").round(0))'''),
    (C, '''# Italian cities: World Bank Subnational Business Ready 2025 (13 cities; five topics; scores 0-100)
br = pd.read_parquet(PROCESSED / "italy_bready_cities.parquet")
t = br[br.level == "topic"].pivot(index="city", columns="topic", values="score"); t["average"] = t.mean(axis=1); t = t.sort_values("average")
area = br[["city", "area"]].drop_duplicates().set_index("city").area
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
vals = t.drop(columns="average")
axes[0].imshow(vals.values, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("s", ["#f0efec", "#2a78d6"]), vmin=50, vmax=90, aspect="auto"); axes[0].grid(False)
axes[0].set_xticks(range(5), [c.replace("Business ", "").replace(" Services", "") for c in vals.columns], rotation=20, fontsize=8); axes[0].set_yticks(range(len(t)), [f"{c} ({area[c]})" for c in t.index], fontsize=8)
for i in range(vals.shape[0]):
    for j in range(5): axes[0].text(j, i, f"{vals.values[i, j]:.0f}", ha="center", va="center", fontsize=7, color=INK)
axes[0].set_title("City scores by topic (0-100)")
pl = br[br.level == "pillar"].groupby("item").score.agg(["mean", "min", "max"]).loc[["Regulatory Framework", "Public Services", "Operational Efficiency"]]
axes[1].barh(range(3), pl["mean"], color=ITALY_C, height=0.5); axes[1].errorbar(pl["mean"], range(3), xerr=[pl["mean"] - pl["min"], pl["max"] - pl["mean"]], fmt="none", ecolor=INK2, capsize=3)
axes[1].set_yticks(range(3), pl.index); axes[1].invert_yaxis(); axes[1].set_xlim(40, 100); axes[1].set_title("Pillar scores: mean and range across cities"); fig.tight_layout(); save(fig, "09_bready_cities")
display(t.round(1)); print("average by macro-area:", br[br.level == "topic"].groupby("area").score.mean().round(1).to_dict()); print("pillar means:", pl["mean"].round(1).to_dict())'''),
    (C, '''# EU cohesion funds 2014-2020 (ERDF, ESF, Cohesion Fund, YEI): spending declared vs programmed, by member state
ab = pd.read_parquet(PROCESSED / "esif_absorption.parquet"); ab = ab[ab.fund.isin(["ERDF", "ESF", "CF", "YEI"])]
from cifresignificative.countries import EU27 as EU27_GEO
ab = ab[ab.geo.isin(EU27_GEO)].groupby("geo")[["planned", "spent"]].sum(); ab["absorption"] = ab.spent / ab.planned * 100; ab = ab.sort_values("absorption")
fig, ax = plt.subplots(figsize=(8.5, 5.2)); cols = [ITALY_C if g == "IT" else PEER_C for g in ab.index]
ax.barh(range(len(ab)), ab.absorption, color=cols, height=0.65); ax.axvline(100, color=INK2, lw=1, ls=":"); ax.axvline(ab.drop("IT").absorption.median(), color=EU_C, lw=1.6, ls="--")
ax.set_yticks(range(len(ab)), [f"{g}" for g in ab.index], fontsize=8); ax.set_xlabel("Expenditure declared / programmed amount, %"); ax.set_title("EU cohesion funds 2014-2020: absorption (dashed = median of others)"); save(fig, "09_esif_absorption")
print("Italy: %.1f%% of EUR %.1fbn programmed | median of others %.1f%% | rank among %d (1 = lowest): %d | countries above 100%%: %d" % (ab.loc["IT", "absorption"], ab.loc["IT", "planned"] / 1e9, ab.drop("IT").absorption.median(), len(ab), list(ab.index).index("IT") + 1, int((ab.absorption > 100).sum())))'''),
    (C, '''# Italy only: Istat non-observed economy (underground + illegal), % of GDP, and underground share of value added by industry
from cifresignificative.sources import istat_unobserved
nz = istat_unobserved.load_components().pivot(index="year", columns="component", values="pct_gdp")
ind = istat_unobserved.load_by_industry(); ind = ind[(ind.component == "underground") & (ind.year == 2023)].set_index("industry").pct_value_added.drop("Totale").sort_values()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]; ax.stackplot(nz.index, nz.underreporting, nz.irregular_labour, nz.other_underground, nz.illegal, colors=[ITALY_C, EU_C, PEER_C, INK2], labels=["Under-reporting", "Irregular labour", "Other underground", "Illegal activities"], alpha=0.9)
ax.set_ylabel("% of GDP"); ax.set_title("Non-observed economy, Italy"); ax.legend(loc="lower left", fontsize=8, frameon=False)
ax = axes[1]; ax.barh(range(len(ind)), ind.values, color=ITALY_C, height=0.65); ax.set_yticks(range(len(ind)), [i[:32] for i in ind.index], fontsize=8); ax.set_xlabel("Underground economy, % of industry value added (2023)"); ax.set_title("By industry")
fig.tight_layout(); save(fig, "09_istat_unobserved")
print("Non-observed economy: %.1f%% of GDP in 2011, peak %.1f%% in %d, %.1f%% in 2023 (EUR %.0fbn)" % (nz.non_observed_total[2011], nz.non_observed_total.max(), nz.non_observed_total.idxmax(), nz.non_observed_total[2023], istat_unobserved.load_components().query("component == 'non_observed_total' and year == 2023").meur.iloc[0] / 1000))'''),
    (M, """## Italy-only evidence from primary sources (not comparable with the EU VAT-gap series)
- **Ministry of Economy and Finance, 2025 report on evasion** (tables I.1, I.4, I.5; upper estimates): the gap in tax revenue was EUR 95.6bn in 2018 and EUR 90.9bn in 2022 (nominal), or 19.7% and 17.0% of tax due (20.1% to 17.2% excluding IMU and excise duties; the report notes this is a third of the 2004 peak of 38.3% and 11.7% below 2019). Including social contributions the gap was EUR 108.4bn (2018) and EUR 102.5bn (2022); the contributions part fell from EUR 12.8bn to 11.6bn.
- **By tax (2022):** self-employed and business income tax (IRPEF) EUR 37.0bn, a propensity of 59.8% (60.2% in 2018); VAT EUR 29.0bn, 18.4% of VAT due (23.4% in 2018; the ministry's method differs from the Commission's 15.0% in 2023); IRES EUR 10.3bn; IRAP EUR 3.6bn. The recent decline is concentrated in VAT, IRAP and IRES, not in self-employed income tax.
- **Istat irregular-employment rate** (share of work performed outside regulation, latest edition): 10.8% in 2005 and 10.0% in 2023; the code for the 'total' status is inferred (see docs/data_dictionary.md).
- **Business environment in 13 Italian cities (World Bank Subnational Business Ready 2025, five topics):** averages run from 72.1 (Reggio Calabria) to 76.6 (Ancona) with no North-South gradient (North 75.1, Centre 76.0, South 74.3, Islands 75.1). The weak spots are the same everywhere: dispute resolution (52.5 in Reggio Calabria to 65.7 in Genoa) and, by pillar, operational efficiency (mean 64.2, against 77.9 for public services and 83.3 for the regulatory framework): the rules are largely national and uniform; what differs and what lags is how they work in practice. Business entry scores about 84 in every city.
- **EU cohesion funds (2014-2020 ERDF, ESF, Cohesion Fund, YEI):** Italy declared 95.7% of the EUR 64.4bn programmed, the 4th lowest of the EU27 (Ireland 79.2%, Sweden 90.0%, Luxembourg 95.3%; median of the others 110.7%); 21 of 26 countries are above 100% because programmes were over-committed. This measures spending against programming at the final reporting date, not the quality of projects or timeliness (the data do not show delays within the period). Late-payment times of public administrations are not covered (no harmonised official series found).
- **Istat non-observed economy (2025 release, years 2011-2023):** the press figure is confirmed: 10.1% of GDP in 2022 (EUR 202bn) and 10.2% in 2023 (EUR 218bn), made of underground economy 9.2% (under-reporting 5.0%, irregular labour 3.6%, other 0.6%) and illegal activities 0.9%. The peak was 12.5% in 2014; the level fell to 9.0% for the underground part in 2020 and has not declined further since. Relative to value added the underground economy is largest in other personal services (32.4%), trade, transport, accommodation and food (18.8%) and construction (16.5%)."""),
]

NOTEBOOKS["08_productivity_sectors"].append((M, """## Takeaways (Eurostat national accounts; market sectors excl. real estate; 2005-2025, 2025 a first estimate, data of 21 October 2026)
- **Aggregate output per hour barely moved:** index 2005 = 100 reaches 102 in 2025 for Italy against 127 for the median of the other members (Germany 114, France 110, Spain 115).
- **The gap is within sectors, not between them.** Over 2005-2025 within-sector productivity added +2.5% for Italy vs +21.9% for the median of the other 24 members of the sample (Germany +14.6, France +9.3, Spain +16.4; Italy is 3rd lowest of 25); the reallocation of hours across the nine broad sections was +0.1 for Italy vs +3.4 median, the interaction -1.2 vs -1.0, and the total change +1.5% vs +26.8% (median of the totals: the medians of the parts do not add up). A shift of hours between sections is therefore not what explains the gap. The sample is the 25 members with all nine sections in 2005 and 2025 (Bulgaria and Latvia have no 2025 value added yet). For 2005-2019 the same 25 members give +2.8 vs +14.3 within sectors, -0.5 vs +2.1 reallocation (all 27 members: +16.2 and +2.4). This view cannot see misallocation across *firms within* a section, which is where the literature (Hsieh-Klenow; Gopinath et al.) places it: that needs firm-level data.
- **Italy trails the median in all nine sections over 2005-2025** (seven of nine over 2005-2019). Annual growth in output per hour: agriculture +0.15% (median +2.4%), information & communication +0.9% (+1.8%), industry +0.6% (+1.5%), finance +0.65% (+1.3%), trade, transport & hospitality +0.7% (+1.1%), professional & administrative services -0.6% (+0.8%), construction -0.3% (+0.1%), public administration, education and health -0.8% (0.0%), arts and other services -0.1% (+0.1%). **Construction is period-dependent:** -1.4% a year against +0.4% in 2005-2019, but +2.3% a year against -1.3% in 2019-2025, a period that contains the building-bonus years (a coincidence in time, not tested; construction value added also carries measurement questions).
- Structure: industry fell from 20.6% to 17.0% of hours (2005-2025) while professional & administrative services rose from 9.9% to 13.6%, a sector where measured productivity fell. Deflators in services are harder to measure, so sector rankings should be treated with care."""))

NOTEBOOKS["09_governance_evasion"].append((M, """## Takeaways
- **Governance scores sit below the northern peers and near the southern ones.** In 2024 Italy scores 69 on government effectiveness (2005: 61; median of the other members 75), 59 on control of corruption (unchanged since 2005; Germany 84, France 73, Spain 63), 67 on regulatory quality (2005: 72) and 68 on rule of law. These are perception-based composites, recalculated under the World Bank's 2025 methodology.
- **VAT compliance gap:** Italy's fell from 27.6% of VAT due in 2013 to 15.0% in 2023, the 5th highest of 26 countries with data, versus an 8.0% median (Germany 9.7%, France 5.6%, Spain 7.6%, Romania 30.0%). The improvement coincides with electronic invoicing, but this series does not prove the cause.
- Across the EU27 the VAT gap is only moderately related to control of corruption (correlation -0.35 in 2023), and government effectiveness tracks income closely (0.80, driven partly by Luxembourg and Ireland). These are cross-sections, not causal tests.
- **Public procurement (2021-2024):** Italy's share of single-bidder contracts was 34% in 2024 against a 27% median of the other members (10th highest of 26; 37% in 2022-23); direct awards are low (3% vs 5%), but decisions take 123 days vs a 74-day median (4th slowest of 27; 172 days in 2023).
- **Not yet available:** a harmonised income/corporate tax gap and any direct measure of spending waste."""))

NOTEBOOKS["10_spending_composition"] = [
    (M, "# 10 - Where does the public money go?\nGeneral-government expenditure by function (COFOG) and by economic nature, % of GDP. There is no official measure of 'bad spending': this notebook shows **composition** and where Italy departs from the other members. 2021-23 are distorted by one-off items (see the capital-transfer chart)."),
    (C, '''sp = load_spending()
items = [
    (spend_wide(sp, "TE", "nature"), "Total expenditure", "% of GDP"),
    (spend_wide(sp, "TR", "nature"), "Total revenue", "% of GDP"),
    (spend_wide(sp, "GF1002", "cofog"), "Old-age pensions & benefits", "% of GDP"),
    (spend_wide(sp, "GF0107", "cofog"), "Interest on public debt", "% of GDP"),
    (spend_wide(sp, "D9PAY", "nature"), "Capital transfers", "% of GDP"),
    (spend_wide(sp, "D1PAY", "nature"), "Public sector wage bill", "% of GDP"),
]
small_multiples_wide(items, ncols=3, name="10_spending_overview")'''),
    (C, '''# Composition gap: Italy minus the median of the other members, by function, % of GDP
def gap(year):
    t = cofog_groups(sp, year); eu = [c for c in t.index if c in EU27_ISO3]
    return (t.loc[ITALY] - t.loc[[c for c in eu if c != ITALY]].median())
g19, g24 = gap(2019), gap(2024)
order = g24.sort_values().index
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, g, yr in zip(axes, [g19, g24], [2019, 2024]):
    ax.barh(range(len(order)), g[order].values, color=[ITALY_C if v >= 0 else EU_C for v in g[order].values], height=0.6)
    ax.set_yticks(range(len(order)), order); ax.axvline(0, color=INK2, lw=0.8); ax.set_title(f"{yr}: Italy minus median of the other EU27"); ax.set_xlabel("pp of GDP")
    for i, v in enumerate(g[order].values): ax.annotate(f"{v:+.1f}", (v, i), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points", ha="left" if v >= 0 else "right", va="center", fontsize=7, color=INK2)
fig.suptitle("Blue = Italy spends more than the median, orange = less", fontsize=9, color=INK2, x=0.01, ha="left"); fig.tight_layout(); save(fig, "10_cofog_gap")
tot = {y: spend_wide(sp, "TOTAL", "cofog").pipe(lambda w: (w.loc[y, ITALY], eu_median_ex_italy(w).loc[y])) for y in (2005, 2019, 2024)}
print("Total expenditure, Italy vs median of others (% GDP):", {y: (round(a, 1), round(b, 1)) for y, (a, b) in tot.items()})
display(pd.DataFrame({"2019": g19, "2024": g24}).loc[order].round(1))'''),
    (C, '''# The same gap by economic nature (not exhaustive: other items are omitted)
def ngap(year):
    rows = {}
    for it, lab in NATURE_LABELS.items():
        w = spend_wide(sp, it, "nature"); rows[lab] = w.loc[year, ITALY] - eu_median_ex_italy(w).loc[year]
    return pd.Series(rows)
n19, n24 = ngap(2019), ngap(2024)
order = n24.sort_values().index
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, g, yr in zip(axes, [n19, n24], [2019, 2024]):
    ax.barh(range(len(order)), g[order].values, color=[ITALY_C if v >= 0 else EU_C for v in g[order].values], height=0.6)
    ax.set_yticks(range(len(order)), order); ax.axvline(0, color=INK2, lw=0.8); ax.set_title(f"{yr}: Italy minus median of the other EU27"); ax.set_xlabel("pp of GDP")
fig.tight_layout(); save(fig, "10_nature_gap")
rev = {k: (spend_wide(sp, k, "nature")[ITALY].loc[2019], eu_median_ex_italy(spend_wide(sp, k, "nature")).loc[2019]) for k in ["D5REC", "D61REC", "D2REC", "TR"]}
print("Revenue 2019, Italy vs median (% GDP): direct taxes", [round(x, 1) for x in rev["D5REC"]], "| social contributions", [round(x, 1) for x in rev["D61REC"]], "| indirect taxes", [round(x, 1) for x in rev["D2REC"]], "| total", [round(x, 1) for x in rev["TR"]])
display(pd.DataFrame({"2019": n19, "2024": n24}).loc[order].round(1))'''),
    (C, '''# Primary spending (total expenditure minus interest) vs the median, over time
te = spend_wide(sp, "TE", "nature"); it = spend_wide(sp, "D41PAY", "nature"); prim = te - it
fig, ax = plt.subplots(figsize=(7.5, 3.8)); line_wide(ax, prim, "Primary expenditure (excl. interest)", "% of GDP"); save(fig, "10_primary_spending")
print("Primary expenditure Italy vs median:", {y: (round(prim.loc[y, ITALY], 1), round(eu_median_ex_italy(prim).loc[y], 1)) for y in (2005, 2010, 2015, 2019, 2023, 2024)})'''),
]

NOTEBOOKS["11_wages_gender"] = [
    (M, "# 11 - Wages and gender equality\nEurostat series on pay, working patterns, and gender gaps. Note: Italy has **no statutory national minimum wage**, so the Eurostat minimum-wage table has no Italian value."),
    (C, '''emp_gap = wide(panel, "employment_rate_20_64_male") - wide(panel, "employment_rate_20_64_female")
items = [
    (emp_gap, "Employment gap, men minus women (20-64)", "pp"),
    (wide(panel, "part_time_female_20_64"), "Women in part-time work (20-64)", "% of employed women"),
    (wide(panel, "involuntary_part_time_female_15_64"), "Involuntary part-time, women", "% of part-time women"),
    (wide(panel, "inactive_care_female_25_54"), "Women outside labour force for care (25-54)", "% of women 25-54"),
    (wide(panel, "women_managers_share_pct"), "Women among managers", "% of managers"),
    (wide(panel, "gender_pension_gap_65"), "Gender pension gap (65+)", "% lower for women"),
]
small_multiples_wide(items, ncols=3, name="11_gender_overview")
latest = {n: (w[ITALY].dropna().index.max(), round(w[ITALY].dropna().iloc[-1], 1), round(eu_median_ex_italy(w).loc[w[ITALY].dropna().index.max()], 1), "%d/%d" % italy_rank(w, w[ITALY].dropna().index.max())) for w, n, _ in items}
display(pd.DataFrame(latest, index=["year", "Italy", "median others", "rank (1 = highest)"]).T)'''),
    (C, '''# Pay gaps: unadjusted vs overall. The unadjusted hourly gap is low where few women work (selection effect).
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
line_wide(axes[0], wide(panel, "gender_pay_gap"), "Unadjusted gender pay gap (hourly)", "% of men's pay")
line_wide(axes[1], wide(panel, "gender_overall_earnings_gap"), "Overall earnings gap (pay x hours x employment)", "%")
x = wide(panel, "employment_rate_20_64_female", EU27_ISO3).loc[2022]; y = wide(panel, "gender_pay_gap", EU27_ISO3).loc[2022]
scatter_italy(axes[2], x, y, "Female employment rate 20-64, 2022", "Unadjusted gender pay gap, %", "Fewer women employed, smaller measured pay gap")
fig.tight_layout(); save(fig, "11_pay_gaps")
w = wide(panel, "gender_overall_earnings_gap"); yl = w[ITALY].dropna().index.max()
print("Overall earnings gap", yl, "Italy:", round(w.loc[yl, ITALY], 1), "| median of others:", round(eu_median_ex_italy(w).loc[yl], 1), "| rank", italy_rank(w, yl))
print("corr(female employment, unadjusted pay gap) across EU27, 2022:", round(x.corr(y), 2))'''),
    (C, '''# Earnings level (average worker, single, no children, PPS): Italy relative to the median of the other members, and to DEU/FRA/ESP
net = wide(panel, "net_earnings_pps"); gross = wide(panel, "gross_earnings_pps")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, w, t in zip(axes, [gross, net], ["Gross annual earnings", "Net annual earnings"]):
    rel = w.div(eu_median_ex_italy(w), axis=0) * 100
    for c, lab in [("DEU", "Germany"), ("FRA", "France"), ("ESP", "Spain")]:
        d = rel[c].dropna(); ax.plot(d.index, d.values, color=PEER_C, lw=1.4); ax.annotate(lab, (d.index[-1], d.iloc[-1]), xytext=(3, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
    d = rel[ITALY].dropna(); ax.plot(d.index, d.values, color=ITALY_C, lw=2.5); ax.annotate("Italy", (d.index[-1], d.iloc[-1]), xytext=(3, 0), textcoords="offset points", color=ITALY_C, fontweight="bold", va="center")
    ax.axhline(100, color=EU_C, ls="--", lw=1.5); ax.set_title(f"{t}, PPS (median of others = 100)")
fig.tight_layout(); save(fig, "11_earnings_relative")
for lab, w in [("gross", gross), ("net", net)]:
    rel = (w[ITALY] / eu_median_ex_italy(w) * 100).dropna(); print(lab, "Italy vs median of others:", {y: round(rel.loc[y], 0) for y in (2005, 2010, 2015, 2019, 2023) if y in rel.index}, "| vs Germany:", {y: round(w.loc[y, ITALY] / w.loc[y, "DEU"] * 100, 0) for y in (2005, 2023)})'''),
    (C, '''# Labour cost, tax wedge, in-work poverty
items = [
    (wide(panel, "tax_wedge_low_wage"), "Tax wedge, low-wage earners", "%"),
    (wide(panel, "labour_cost_hourly_eur"), "Hourly labour cost (biennial before 2020)", "EUR"),
    (wide(panel, "inwork_poverty_18_64"), "In-work poverty (18-64)", "% of workers"),
]
small_multiples_wide(items, ncols=3, name="11_wages_other")'''),
]

NOTEBOOKS["10_spending_composition"].append((M, "## Absolute amounts: Italy in million euro\nEurostat also publishes COFOG in million euro at current prices. Euro levels are shown for Italy only (they are not comparable across countries of different size); the cross-country comparison stays in % of GDP and % of total spending. Nominal growth includes inflation."))
NOTEBOOKS["10_spending_composition"].append((C, '''yrs = [2005, 2019, 2024]
eur = italy_cofog_euro(sp, yrs)
eur["nominal growth 2005-24, %"] = 100 * (eur["2024 MEUR"] / eur["2005 MEUR"] - 1)
display(eur.sort_values("2024 MEUR", ascending=False).round(1))
tot = eur[[f"{y} MEUR" for y in yrs]].sum(); print("Total (MEUR):", tot.round(0).to_dict(), "| nominal growth 2005-24: %.0f%%" % (100 * (tot["2024 MEUR"] / tot["2005 MEUR"] - 1)))'''))
NOTEBOOKS["10_spending_composition"].append((C, '''# Defence: where Italy ranks among the EU27 (descriptive placebo ranking over countries, no test), both units, 2024
for unit, lab in [("PC_GDP", "% of GDP"), ("PC_TOT", "% of total spending")]:
    w = spend_wide(sp, "GF02", unit=unit); r = w.loc[2024, [c for c in w.columns if c in EU27_ISO3]].dropna()
    print(f"Defence {lab}, 2024: Italy {r[ITALY]:.1f}, median of others {r.drop(ITALY).median():.1f}, rank {int(r.rank(ascending=False)[ITALY])}/{len(r)} (1 = highest)")
w = spend_wide(sp, "GF02", unit="MIO_EUR"); print("Italy defence, MEUR:", w[ITALY].dropna().loc[[2005, 2019, 2024]].round(0).to_dict(), "(COFOG covers the Defence function only: NATO-definition totals differ, not checked here)")'''))
NOTEBOOKS["10_spending_composition"].append((M, """## Takeaways (Eurostat COFOG and national-accounts items; median = other EU27 members)
- **Italy spends more overall:** total expenditure was 47.1% of GDP in 2005, 48.4% in 2019 and 50.4% in 2024, against medians of 44.1%, 42.3% and 46.6%.
- **The excess is concentrated in three items.** In 2019 old-age pensions were +5.2 pp of GDP above the median, debt interest +2.2 and survivors' pensions +1.7 (2024: +4.3, +2.6, +1.5). Almost every other function is within about 1 pp of the median: education -0.8 (-1.0 in 2024), other social protection -0.7 (-0.9), economic affairs -0.6 (-0.5), health +0.8 (-0.1).
- **By economic nature (2024):** social benefits +5.1 pp, interest +2.6, capital transfers +0.8; the public-sector wage bill is 2.3 pp *below* the median (1.1 below in 2019), public investment 0.6 below (1.2 in 2019), purchases of goods and services 0.4 below. These aggregates show no sign of an oversized administration; they cannot show whether the money is spent well.
- **Primary spending** (excluding interest) was 45.1% of GDP in 2019 vs 40.7% median (+4.4 pp) and 46.5% vs 44.4% in 2024 (+2.1).
- **Revenue is also high** (2019: 47.0% of GDP vs 43.2%), mostly direct taxes (14.3% vs 9.4%) and social contributions (13.4% vs 12.0%); indirect taxes are at the median (14.5% vs 14.1%).
- **2021-23 are distorted.** Housing and community spending jumped from 0.4% of GDP (2019) to 2.4%, 3.3% and 4.4%, and capital transfers from 1.0% to 5.0%, 5.1% and 5.9%, before falling back in 2024 (0.8% and 1.7%). This is consistent with the building tax-credit schemes being recorded as spending, but the data do not name the scheme.
- **In euro (Italy only, current prices):** total general-government spending rose from EUR 705.4bn (2005) to 873.6bn (2019) and 1,109.2bn (2024), +57% nominal over 2005-24 (inflation included). Old-age pensions went from 174.6bn to 306.4bn (+75%; 27.6% of the 2024 total), health from 101.7bn to 146.1bn (+44%), debt interest from 70.8bn to 87.8bn (+24%). The 2024 euro figures reproduce the % of total spending in the table above.
- **Defence:** EUR 18.9bn (2005), 23.1bn (2019), 28.3bn (2024; +50%), 1.3% of GDP and 2.6% of total spending in 2024 against medians of 1.6% and 3.2% (19th of 27 and 20th of 27, 1 = highest; a descriptive ranking, no test). COFOG covers the Defence function only; NATO-definition totals differ and were not checked here. As elsewhere, descriptive, not causal."""))

NOTEBOOKS["11_wages_gender"].append((M, """## Takeaways (Eurostat; 2025 values are provisional)
- **Employment is the core gender gap:** men's employment rate (20-64) exceeds women's by 19.1 pp against a 7.0 pp median, the largest in the EU27 (it was 26 pp in 2005). Italy's gap in women outside the labour force for care reasons (25-54) is 1.4% against 0.7% (4th of 19), down from 4.3% in 2006.
- **Women who work are often not working the hours they want:** 27.7% of employed women work part time (median 14.8%), and 46.6% of those part-timers want full-time work (median 18.0%; highest of 25 countries); for part-time men it is 65.9% (2nd of 24).
- **The unadjusted pay gap hides the problem.** Italy's hourly gap is about 5% (2024, provisional; 2.2% in 2023) against an 11.5% median, but the *overall* earnings gap, which combines pay, hours and employment, was 43.0% in 2018 (median 27.7%; 3rd highest; the latest year available). Across the EU27 the unadjusted gap is positively correlated with female employment (+0.49 in 2022): where fewer women work, the measured gap looks smaller.
- **Representation and pensions:** women are 29.0% of managers (median 37.5%; 25th of 26; there is a series break in 2011) and the gender pension gap is 29.2% (median 22.0%; 5th highest).
- **Pay levels:** for an average single worker, gross earnings in PPS were 120 and net earnings 114 (median of the other members = 100) in 2023, up from 106 and 95 in 2005 but down from the 2019 peak (129 and 107). Against Germany, gross pay was 64% in 2005 and 63% in 2023, net 80% then 72%. Hourly labour cost was EUR 31.0 in 2024 vs 20.3 (median), 43.5 (Germany), 43.4 (France). 2024 earnings are excluded (source anomaly).
- **In-work poverty** is 10.3% vs 8.3% (7th highest); the tax wedge on low earners is 38.7% vs 37.4% (11th). Italy has no statutory national minimum wage (no value in the Eurostat table)."""))

NOTEBOOKS["12_higher_education"] = [
    (M, "# 12 - Higher education: attainment, fields (STEM), gender, doctorates, age\nEurostat education statistics (UOE data collection, EU Labour Force Survey). STEM = natural sciences, mathematics & statistics (F05), ICT (F06), engineering, manufacturing & construction (F07). Graduate counts are degrees, not people: a student with a bachelor and a master counts twice, and countries differ in cycle structure."),
    (C, '''gender_gap = wide(panel, "tertiary_attainment_25_34_female") - wide(panel, "tertiary_attainment_25_34_male")
items = [
    (wide(panel, "tertiary_attainment_25_34"), "Tertiary attainment, 25-34", "% of age group"),
    (gender_gap, "Women minus men, tertiary attainment 25-34", "pp"),
    (wide(panel, "tertiary_spend_per_student_pps"), "Public spending per tertiary student", "PPS"),
    (wide(panel, "stem_share_grads_pct"), "STEM share of tertiary graduates", "% of graduates"),
    (wide(panel, "women_share_stem_grads_pct"), "Women among STEM graduates", "%"),
    (wide(panel, "phd_grads_per_1000_pop_25_34"), "Doctoral graduates per 1000 aged 25-34", "per 1000"),
    (wide(panel, "mean_age_grads_bachelor"), "Mean age of bachelor graduates", "years"),
    (wide(panel, "bachelor_grads_30plus_pct"), "Bachelor graduates aged 30+", "%"),
    (wide(panel, "recent_grads_employment_rate"), "Employment of recent graduates (1-3 years out)", "% of graduates aged 20-34"),
]
small_multiples_wide(items, ncols=3, name="12_higher_ed_overview")'''),
    (C, '''he = load_highered(); Y = broad_year(he, "he_grads_field"); print("Field analysis year:", Y)
sh = field_shares(he, "he_grads_field", "ED5-8", "T", Y); eu = [c for c in sh.index if c in EU27_ISO3]
tab = pd.DataFrame({"Italy": sh.loc[ITALY], "Median of other EU27": sh.loc[[c for c in eu if c != ITALY]].median(), "Germany": sh.loc["DEU"]}).rename(index=FIELD_LABELS).sort_values("Italy")
fig, ax = plt.subplots(figsize=(8.5, 4.6)); yy = np.arange(len(tab))
ax.barh(yy, tab["Italy"], color=ITALY_C, height=0.5, label="Italy", zorder=2)
ax.scatter(tab["Median of other EU27"], yy, color=EU_C, s=42, marker="D", zorder=3, label="median of other EU27")
ax.scatter(tab["Germany"], yy, color=PEER_C, s=36, zorder=2, label="Germany")
ax.set_yticks(yy, tab.index); ax.set_xlabel("% of tertiary graduates"); ax.set_title(f"What graduates study, {Y}"); ax.legend(fontsize=8, loc="lower right"); save(fig, "12_fields")
display(tab.round(1)); print("STEM share (F05+F06+F07):", {k: round(v, 1) for k, v in {"Italy": sh.loc[ITALY, STEM_FIELDS].sum(), "median others": sh.loc[[c for c in eu if c != ITALY], STEM_FIELDS].sum(axis=1).median(), "Germany": sh.loc["DEU", STEM_FIELDS].sum()}.items()})'''),
    (C, '''# Women's share of graduates by field, and doctoral level
wf = women_share_by_field(he, "he_grads_field", "ED5-8", Y); wd = women_share_by_field(he, "he_grads_field", "ED8", Y)
rows = []
for lab, w in [("all tertiary graduates", wf), ("doctoral graduates", wd)]:
    eu = [c for c in w.index if c in EU27_ISO3 and c != ITALY]
    rows.append(pd.DataFrame({"Italy": w.loc[ITALY], "median of other EU27": w.loc[eu].median()}).rename(index={**FIELD_LABELS, "TOTAL": "All fields"}).assign(level=lab))
t = pd.concat(rows).reset_index(names="field")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, (lab, g) in zip(axes, t.groupby("level", sort=False)):
    g = g.set_index("field").loc[[*FIELD_LABELS.values(), "All fields"]]; yy = np.arange(len(g))
    ax.barh(yy, g["Italy"], color=ITALY_C, height=0.5, zorder=2, label="Italy"); ax.scatter(g["median of other EU27"], yy, color=EU_C, marker="D", s=40, zorder=3, label="median of other EU27")
    ax.axvline(50, color=INK2, lw=0.8, ls=":"); ax.set_yticks(yy, g.index); ax.set_title(f"Women as % of {lab}, {Y}"); ax.set_xlabel("% women")
axes[0].invert_yaxis(); axes[0].legend(fontsize=8, loc="lower right"); fig.tight_layout(); save(fig, "12_women_by_field")
display(t.pivot(index="field", columns="level", values=["Italy", "median of other EU27"]).round(0))'''),
    (C, '''# Age at graduation: share of bachelor and master graduates by single year of age
ages = [f"Y{a}" for a in range(20, 36)]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, (lvl, nm) in zip(axes, [("ED6", "Bachelor"), ("ED7", "Master")]):
    ya = broad_year(he[he.isced11 == lvl], "he_grads_age")
    p = age_profile(he, lvl, ya, ages); oth = p.loc[[c for c in p.index if c in EU27_ISO3 and c != ITALY]]
    x = [int(a[1:]) for a in ages]
    ax.fill_between(x, oth.quantile(0.25), oth.quantile(0.75), color=PEER_C, alpha=0.4, label="middle half of other EU27")
    ax.plot(x, oth.median(), color=EU_C, ls="--", lw=2, label="median of other EU27"); ax.plot(x, p.loc[ITALY], color=ITALY_C, lw=2.6, label="Italy")
    ax.set_title(f"{nm} graduates by age, {ya}"); ax.set_xlabel("age")
axes[0].set_ylabel("% of graduates"); axes[0].legend(fontsize=8); fig.tight_layout(); save(fig, "12_graduation_age")'''),
    (C, '''# Latest values and rank among the EU27 for all higher-education indicators
names = ["tertiary_attainment_25_34", "tertiary_attainment_25_34_female", "tertiary_attainment_25_34_male", "tertiary_attainment_25_64", "tertiary_spend_per_student_pps",
         "grads_per_1000_pop_20_29", "stem_grads_per_1000_pop_20_29", "stem_share_grads_pct", "stem_share_grads_female_pct", "stem_share_grads_male_pct", "women_share_stem_grads_pct",
         "women_share_ict_grads_pct", "ict_share_grads_pct", "women_share_grads_pct", "phd_grads_per_1000_pop_25_34", "phd_women_share_pct", "phd_stem_share_pct",
         "mean_age_grads_bachelor", "mean_age_grads_master", "mean_age_grads_phd", "bachelor_grads_30plus_pct", "recent_grads_employment_rate", "recent_grads_employment_rate_female",
         "recent_secondary_grads_employment_rate"]
rows = {}
for n in names:
    w = wide(panel, n); y = int(w[ITALY].dropna().index.max()); y = min(y, 2024)
    rows[n] = {"year": y, "Italy": round(w.loc[y, ITALY], 1), "median others": round(eu_median_ex_italy(w).loc[y], 1), "rank (1=highest)": "%d/%d" % italy_rank(w, y)}
display(pd.DataFrame(rows).T)'''),
]

NOTEBOOKS["12_higher_education"].append((M, """## Takeaways (Eurostat; 2024 unless stated; median = other EU27 members)
- **Attainment is the gap, not graduate output.** 31.6% of 25-34 year-olds hold a tertiary degree (median 44.8%; 26th of 27); women 38.5% (median 55.2%), men 25.0% (median 36.8%). Italian women out-attain men by 13.5 pp, but both are far below peers. Public spending per tertiary student is 8,817 PPS (2023) vs 10,109 (18th of 25).
- **Degree counts look high, which reflects the 3+2 structure:** 91.2 graduates per 1000 people aged 20-29 (median 82.3; 8th) count degrees, not people (bachelor and master both count).
- **STEM is average in size, different in mix.** STEM is 23.2% of graduates (median 24.4%; Germany 35.8%), but ICT is only 1.6% (median 5.7%, last of 27) while natural sciences are 7.9% (median 5.0%) and engineering 13.7% (median 12.0%). Arts & humanities (16.9% vs 8.9%) and social sciences (15.2% vs 9.9%) are over-represented; business (19.4% vs 24.4%), education (6.1% vs 9.6%) and health (12.2% vs 15.0%) under-represented.
- **Gender and STEM.** Women are 58.1% of graduates (median 59.6%) but 39.5% of STEM graduates (median 34.2%; 4th highest), driven by natural sciences (58% women) and engineering (31%); in ICT they are 18% (median 21.5%). 15.8% of women graduates and 33.5% of men graduates study STEM (medians 14.3% and 38.5%).
- **Doctorates.** 1.6 doctoral graduates per 1000 aged 25-34 (median 2.0; 20th of 26); women are 48.1% of PhDs (median 51.5%); 55.3% of PhDs are in STEM (median 42.0%; 4th); the mean age at PhD is 32.4 (median 35.6).
- **Age.** Bachelor graduates are 27.7 on average (median 25.6) and 24.1% are 30 or older (median 15.8%, 6th highest); few Italians graduate at 20-21 and more do so in their late twenties. Master graduates average 29.8 (median 28.8).
- **Labour-market return.** 74.3% of recent graduates (1-3 years out, 20-34) are employed (median 86.2%; women 72.7% vs 84.8%), against 59.3% for recent upper-secondary graduates (median 76.0%). The gap to the EU is large for both groups: weaker demand for young workers matters as much as the skills mix.
- Not covered: first-generation or socio-economic background, field-level earnings, and doctoral stock (UK, US, Japan comparisons would need OECD Education at a Glance)."""))

NOTEBOOKS["13_industry_detail"] = [
    (M, "# 13 - Where in the economy is the productivity gap? 62 industries (NACE A*64)\nNotebook 08 showed that Italy's weak productivity is a *within-sector* problem at the 9-section level. Here the same accounting is done for 62 industries (market sectors; real estate and non-market services excluded) to see which industries carry the gap. Real output per hour, chain-linked 2020 euros; only growth is compared across countries."),
    (C, '''s64 = load_sectors64(); A, B = 2005, 2024  # 2025 is not usable at A*64 detail yet (Eurostat release of 21 Oct 2026: Italy has 15 of 57 industries); an industry enters a country's table only with output and hours in both years
pv = s64.pivot_table(index="iso3", columns=["nace_r2", "year"], values="prod"); hv = s64.pivot_table(index="iso3", columns=["nace_r2", "year"], values="hours_ths")
inds = sorted({n for n, _ in pv.columns})
def contributions(iso):
    rows = []
    for n in inds:
        try: p0, p1, h0, h1 = pv.loc[iso, (n, A)], pv.loc[iso, (n, B)], hv.loc[iso, (n, A)], hv.loc[iso, (n, B)]
        except KeyError: continue
        if min(p0, p1, h0, h1) > 0 and np.isfinite([p0, p1, h0, h1]).all(): rows.append((n, p0, p1, h0, h1))
    d = pd.DataFrame(rows, columns=["ind", "p0", "p1", "h0", "h1"]).set_index("ind")
    s0, s1 = d.h0 / d.h0.sum(), d.h1 / d.h1.sum(); base = (s0 * d.p0).sum()
    return pd.DataFrame({"hours_share_2005": s0 * 100, "growth_pa": np.log(d.p1 / d.p0) / (B - A) * 100, "within": s0 * (d.p1 - d.p0) / base * 100, "realloc": (s1 - s0) * d.p0 / base * 100})
eu = [c for c in EU27_ISO3 if c in pv.index and c != ITALY]
ci = contributions(ITALY); cm = {c: contributions(c) for c in eu}; cm = {c: v for c, v in cm.items() if len(v)}  # a member without any industry in both years (Bulgaria, Latvia in 2025) must not count as a zero
print('members with industry data:', len(cm), '| left out:', sorted(set(eu) - set(cm)), '| industries per member (min/median):', min(len(v) for v in cm.values()), int(np.median([len(v) for v in cm.values()])), '| Italy:', len(ci))
med_within = pd.DataFrame({c: v["within"] for c, v in cm.items()}).median(axis=1); med_growth = pd.DataFrame({c: v["growth_pa"] for c, v in cm.items()}).median(axis=1)
tab = ci.join(med_within.rename("within_median_others")).join(med_growth.rename("growth_median_others"))
tab["gap_within"] = tab.within - tab.within_median_others
print(f"Aggregate within-industry contribution 2005-2024 (% of 2005 output per hour): Italy {ci.within.sum():.1f} | sum of medians {med_within.sum():.1f}")
print("Reallocation across the 62 industries: Italy %.1f | median of others %.1f" % (ci.realloc.sum(), pd.Series({c: v.realloc.sum() for c, v in cm.items()}).median()))
worst = tab.sort_values("gap_within").head(10); best = tab.sort_values("gap_within").tail(5)
display(worst.round(2)); print("The 10 industries with the largest shortfall explain %.0f%% of the total shortfall (sum of gaps %.1f pp)" % (worst.gap_within.sum() / tab.gap_within.sum() * 100, tab.gap_within.sum()))
plot = pd.concat([worst, best]).sort_values("gap_within")
fig, ax = plt.subplots(figsize=(8.5, 5.2)); ax.barh(range(len(plot)), plot.gap_within, color=[ITALY_C if v >= 0 else EU_C for v in plot.gap_within], height=0.62)
ax.set_yticks(range(len(plot)), [A64_LABELS.get(n, n) for n in plot.index]); ax.axvline(0, color=INK2, lw=0.8); ax.set_xlabel("Italy minus median of other EU27, pp of 2005 aggregate output per hour")
ax.set_title("Industries carrying the productivity shortfall, 2005-2024 (within-industry contribution)"); save(fig, "13_industry_gap")'''),
    (C, '''# Productivity growth vs hours weight for the biggest Italian industries
big = tab.sort_values("hours_share_2005", ascending=False).head(12)
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.scatter(big.growth_median_others, big.growth_pa, s=big.hours_share_2005 * 40, color=ITALY_C, alpha=0.75, edgecolor=SURFACE)
for n, r in big.iterrows(): ax.annotate(A64_LABELS.get(n, n), (r.growth_median_others, r.growth_pa), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK2)
lim = [min(big.growth_pa.min(), big.growth_median_others.min()) - 0.5, max(big.growth_pa.max(), big.growth_median_others.max()) + 0.5]
ax.plot(lim, lim, color=EU_C, ls="--", lw=1.5); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("median of other EU27, % per year"); ax.set_ylabel("Italy, % per year"); ax.set_title("Output per hour growth, 12 largest Italian industries (bubble = hours share)"); save(fig, "13_industry_scatter")
display(big.round(2))'''),
]

NOTEBOOKS["14_ageing_debt_scenarios"] = [
    (M, "# 14 - Ageing costs and debt scenarios\nEuropean Commission / EPC 2024 Ageing Report projections (2022-2070) and a simple, transparent debt simulation. The simulation is an illustration of arithmetic, not a forecast."),
    (C, '''ar = load_ageing()
pens = ageing_series(ar, "Baseline as % of GDP", "Public pensions, gross"); tot = ageing_series(ar, "Total spending as % of GDP", "Baseline")
oad = ageing_series(ar, "Dependency ratios", "Old-age dependency ratio (3)")
def panel_lines(ax, w, title, ylabel):
    eu = [c for c in EU27_ISO3 if c in w and c != ITALY]
    for c in eu: ax.plot(w.index, w[c], color=PEER_C, lw=0.7, alpha=0.7)
    ax.plot(w.index, w[eu].median(axis=1), color=EU_C, lw=2, ls="--"); ax.plot(w.index, w[ITALY], color=ITALY_C, lw=2.6)
    ax.annotate("Italy", (w.index[-1], w[ITALY].iloc[-1]), xytext=(4, 0), textcoords="offset points", color=ITALY_C, fontweight="bold", va="center"); ax.set_title(title); ax.set_ylabel(ylabel)
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
panel_lines(axes[0], pens, "Public pension spending (baseline)", "% of GDP"); panel_lines(axes[1], tot, "Total ageing-related spending (pensions, health, care, education)", "% of GDP"); panel_lines(axes[2], oad, "Old-age dependency ratio", "65+ per 100 aged 20-64")
fig.tight_layout(); save(fig, "14_ageing_projections")
for nm, w in [("pensions", pens), ("total", tot), ("old-age dependency", oad)]:
    eu = [c for c in EU27_ISO3 if c != ITALY and c in w]  # a few countries are missing some variables in the annex
    print(nm, "| Italy:", {y: round(w.loc[y, ITALY], 1) for y in (2022, 2030, 2040, 2050, 2070)}, "| median of others (n=%d):" % len(eu), {y: round(w.loc[y, eu].median(), 1) for y in (2022, 2030, 2040, 2050, 2070)})'''),
    (C, '''# Sensitivity of Italian pension spending to assumptions (pp of GDP vs baseline; Ageing Report scenarios)
sec = "Public pensions, gross as % of GDP (difference from baseline)"
sens = ar[(ar.iso3 == ITALY) & (ar.section == sec) & ar.year.isin([2040, 2070])].pivot(index="variable", columns="year", values="value")
display(sens.round(2))'''),
    (C, '''from cifresignificative.analysis.scenarios import simulate_debt, stabilising_primary_balance
bal = wide(panel, "gov_balance_pct_gdp")[ITALY]; intr = wide(panel, "gov_interest_pct_gdp")[ITALY]; debt = wide(panel, "gov_debt_pct_gdp")[ITALY]
d0 = debt.loc[2024]; pb0 = (bal + intr).loc[2024]; r0 = intr.loc[2024] / debt.loc[2023]  # both in % of GDP, so the ratio is already a fraction
real = ageing_series(ar, "Macroeconomic assumptions", "Potential GDP (growth rate)")[ITALY] / 100; infl = ageing_series(ar, "Macroeconomic assumptions", "HICP (growth rate)")[ITALY] / 100
r_ar = ageing_series(ar, "Macroeconomic assumptions", "Nominal interest rate")[ITALY] / 100; cost = tot[ITALY]
yrs = range(2025, 2051); g = np.array([(1 + real.loc[y]) * (1 + infl.loc[y]) - 1 for y in yrs])
r = np.array([r0 + (r_ar.loc[y] - r0) * min((y - 2024) / 8, 1) for y in yrs])  # implicit rate converges to the Commission's nominal interest assumption by 2032
drift = np.array([cost.loc[y] - cost.loc[2024] for y in yrs])  # extra ageing-related spending vs 2024, not offset
print(f"2024: debt {d0:.1f}% GDP, primary balance {pb0:+.1f}% GDP, implicit rate {r0*100:.1f}% | assumptions 2025-2050: nominal growth {g.mean()*100:.1f}%, interest rate {r.mean()*100:.1f}%, r-g {np.mean(r-g)*100:+.1f} pp")
scen = {"primary balance stays at 2024 level, ageing costs unoffset": simulate_debt(d0, yrs, pb0, r, g, drift), "primary surplus +2% GDP, ageing costs unoffset": simulate_debt(d0, yrs, 2.0, r, g, drift),
        "primary surplus +3% GDP, ageing costs unoffset": simulate_debt(d0, yrs, 3.0, r, g, drift), "primary surplus +2% GDP, interest rate +1 pp": simulate_debt(d0, yrs, 2.0, r + 0.01, g, drift)}
fig, ax = plt.subplots(figsize=(9, 4.6)); ax.plot(debt.index, debt.values, color=INK2, lw=2)
cols = [EU_C, ITALY_C, "#1baf7a", "#e34948"]
for (lab, sr), col in zip(scen.items(), cols): ax.plot(sr.index, sr.values, color=col, lw=2.2, label=lab)
ax.axhline(60, color=INK2, lw=1, ls=":"); ax.annotate("60% reference", (2040, 62), fontsize=8, color=INK2); ax.set_ylim(50, 195); ax.set_ylabel("gross debt, % of GDP"); ax.set_title("Italy: debt-to-GDP under illustrative scenarios (history in gray)"); ax.legend(fontsize=8, loc="upper left"); save(fig, "14_debt_scenarios")
display(pd.DataFrame({k: v.loc[[2030, 2040, 2050]] for k, v in scen.items()}).round(0).T)
print("Primary surplus that holds debt at %.0f%% of GDP for r-g = 0.5/1.0/1.5/2.0 pp:" % d0, [round(stabilising_primary_balance(d0, 0.5 / 100 + 0.03, 0.03), 1), round(stabilising_primary_balance(d0, 1 / 100 + 0.03, 0.03), 1), round(stabilising_primary_balance(d0, 1.5 / 100 + 0.03, 0.03), 1), round(stabilising_primary_balance(d0, 2 / 100 + 0.03, 0.03), 1)])'''),
]

NOTEBOOKS["13_industry_detail"].append((M, """## Takeaways (Eurostat national accounts, A*64 industries; 2005-2024; 2025 is not usable at this detail yet: Italy has 15 of 57 industries)
- **The shortfall is spread over many industries.** Within-industry productivity growth added 5.8% to Italy's aggregate output per hour over 2005-2024 against 18.1% for the (sum of per-industry) medians of the other 26 members: a gap of 12.3 pp (57 Italian industries, 41 of them below the median). Reallocation of hours across industries added 1.0 vs 3.1 for the median, so shifting workers between industries is not the story.
- **The 10 largest shortfalls account for 69% of the total:** accommodation & food service (-1.5 pp; output per hour -1.6% a year), computers & electronics (-1.2), crop & animal production (-1.0), warehousing & transport support (-0.9), telecommunications (-0.8), construction (-0.8), legal, accounting & head offices (-0.7), travel agencies (-0.6; a very small industry whose output per hour falls 9.6% a year, to be read with care), machinery (-0.6) and other professional & veterinary services (-0.6).
- **Construction is a 2005-2019 story.** Over 2005-2019 it was 2.5 pp of an 11.6 pp shortfall (about a fifth; output per hour -1.4% a year against +0.4%, 9.6% of hours); with 2020-2024 included it is 0.8 pp of 12.3 (-0.4% a year against +0.1%), because Italian construction productivity rises after 2019 (see notebook 08). The 2005-2019 results on the revised data are the same to the first decimal (within-industry 4.2 vs 15.8; ten largest 75%), so the change comes from the period, not from the revisions.
- **Offsets exist:** wholesale trade (+1.6 pp better than the median; +2.7% a year), refined petroleum (+0.4), textiles and clothing (+0.3), financial services (+0.3), basic metals (+0.3).
- Caveats: services deflators and hours are less well measured, so industry rankings are indicative; some members have few industries with data in both years (minimum 4, median 53), which makes their reallocation terms less comparable; the data cannot tell whether low productivity reflects firm size, regulation, skills or measurement, which would need firm-level data."""))

NOTEBOOKS["14_ageing_debt_scenarios"].append((M, """## Takeaways (EC/EPC 2024 Ageing Report baseline; illustrative simulation)
- **Italy's pension bill is high today and keeps rising for a decade.** Gross public pension spending is projected at 15.6% of GDP in 2022, 16.6% in 2030, a peak of 17.1% in 2040, then 15.5% (2050) and 13.7% (2070), against a median of 9.0%, 10.2%, 10.4%, 10.7% and 10.5% for the other members (n=25). Total ageing-related spending (pensions, health, long-term care, education) is 27.3% of GDP in 2022 and 28.3% in 2040, vs 20.8% and 22.8%.
- **Demography:** the old-age dependency ratio is projected to rise from 40.8 to 66.0 (2050) vs 34.7 to 51.1 for the median.
- **Policy sensitivities (pp of GDP, Italian baseline):** a constant retirement age instead of the legislated link to life expectancy would add 0.9 pp by 2040 and 1.2 pp by 2070; 10 pp higher employment of older workers lowers spending by 0.7 pp in 2040; lower migration adds 0.4 pp (2040) and 0.7 pp (2070).
- **Illustrative debt arithmetic** (2024 debt 134.7% of GDP, primary balance +0.5%, implicit rate 2.9% rising to the Commission's interest assumption, nominal growth 3.1%, so r-g about +1.1 pp): keeping the 2024 primary balance while ageing costs are not offset takes debt to 138% (2030), 167% (2040) and 182% (2050). A primary surplus of 2% of GDP gives 128%, 140% and 138%; 3% gives 122%, 122% and 108%. Holding debt at its current level needs a primary surplus of about 1.3% of GDP if r-g is 1.0 pp and 2.6% if it is 2.0 pp.
- The simulation ignores stock-flow adjustments, growth feedbacks from fiscal policy and changes in the interest path beyond the stated rule; use it for orders of magnitude and for the sensitivity to r-g, not as a forecast."""))

NOTEBOOKS["15_spending_efficiency"] = [
    (M, "# 15 - Spending against outcomes\nThe closest feasible answer to 'bad spending': pair what a government spends in an area with an outcome that area is meant to improve, and see where Italy sits relative to what the other EU27 members' relationship predicts. Averages of pre-pandemic years (2015-2019) avoid COVID. Outcomes depend on many things besides public spending (income, lifestyle, demography, private spending), so a residual is a prompt for questions, not a verdict."),
    (C, '''sp = load_spending(); W = slice(2015, 2019)
def avg(w, cols=EU27_ISO3): return w.loc[W, [c for c in cols if c in w]].mean()
health_sp = avg(spend_wide(sp, "GF07", "cofog")); edu_sp = avg(spend_wide(sp, "GF09", "cofog")); pen_sp = avg(spend_wide(sp, "GF1002", "cofog"))
le = avg(wide(panel, "life_expectancy_birth")); tm = avg(wide(panel, "treatable_mortality_rate")); pm = avg(wide(panel, "preventable_mortality_rate")); arop = avg(wide(panel, "arop_65plus"))
pisa = pd.concat([wide(panel, f"wb_pisa_{k}").loc[[2015, 2018]].mean() for k in ("math", "reading", "science")], axis=1).mean(axis=1)[[c for c in EU27_ISO3]]
pairs = [("Health spending vs life expectancy", health_sp, le, "Public health spending, % GDP", "Life expectancy at birth, years", False),
         ("Health spending vs treatable mortality", health_sp, tm, "Public health spending, % GDP", "Treatable deaths per 100,000", True),
         ("Education spending vs PISA", edu_sp, pisa, "Public education spending, % GDP", "PISA mean score (2015, 2018)", False),
         ("Old-age spending vs poverty of the old", pen_sp, arop, "Public old-age spending, % GDP", "At-risk-of-poverty rate, 65+ (%)", True)]
fig, axes = plt.subplots(2, 2, figsize=(11, 8)); rows = []
for ax, (title, x, y, xl, yl, lower_better) in zip(axes.flat, pairs, strict=True):
    scatter_italy(ax, x, y, xl, yl, title)
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna(); o = d.drop(ITALY); b = np.polyfit(o.x, o.y, 1); resid = d.y - np.polyval(b, d.x); z = resid / resid.drop(ITALY).std()
    better = -z if lower_better else z
    rows.append({"pair": title, "Italy x": d.loc[ITALY, "x"], "Italy y": d.loc[ITALY, "y"], "fitted y": np.polyval(b, d.loc[ITALY, "x"]), "residual (SD)": z[ITALY], "outcome rank (1 = best residual)": int(better.rank(ascending=False)[ITALY]), "n": len(d), "corr (others)": o.x.corr(o.y)})
fig.tight_layout(); save(fig, "15_efficiency_scatters"); display(pd.DataFrame(rows).set_index("pair").round(2))'''),
    (C, '''# Outcomes alone: life expectancy and avoidable mortality, Italy vs the median of the other members and large peers
rows = {}
for ind, lab in [("life_expectancy_birth", "Life expectancy at birth (years)"), ("treatable_mortality_rate", "Treatable mortality (per 100k)"), ("preventable_mortality_rate", "Preventable mortality (per 100k)"), ("healthy_life_years_65", "Healthy life years at 65 (self-reported)")]:
    w = wide(panel, ind); y = 2019
    rows[lab] = {"Italy 2019": w.loc[y, ITALY], "median others": eu_median_ex_italy(w).loc[y], "DEU": w.loc[y, "DEU"], "FRA": w.loc[y, "FRA"], "ESP": w.loc[y, "ESP"], "rank (1=highest)": "%d/%d" % italy_rank(w, y)}
display(pd.DataFrame(rows).T.round(1))'''),
]

NOTEBOOKS["16_long_history_counterfactual"] = [
    (M, "# 16 - Long history and a synthetic-control counterfactual (1960-2024)\nAMECO long series in national currency: compare countries through indices and ratios only. The synthetic Italy is a weighted mix of other countries that best reproduces Italy's real GDP per head before 1995; the gap afterwards is a *descriptive* measure of divergence from that peer mix, not a causal effect of any single policy (the donors are themselves exposed to the euro, the single market and global shocks)."),
    (C, '''lh = load_long(); gdppc = long_wide(lh, "gdp_real_per_head_ncu")
idx = gdppc / gdppc.loc[1995] * 100
fig, ax = plt.subplots(figsize=(8.5, 4.4))
for c, lab in [("DEU", "Germany"), ("FRA", "France"), ("ESP", "Spain"), ("GBR", "UK")]:
    ax.plot(idx.index, idx[c], color=PEER_C, lw=1.5); ax.annotate(lab, (2024, idx.loc[2024, c]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
ax.plot(idx.index, idx[ITALY], color=ITALY_C, lw=2.8); ax.annotate("Italy", (2024, idx.loc[2024, ITALY]), xytext=(4, 0), textcoords="offset points", color=ITALY_C, fontweight="bold", va="center")
ax.axvline(1999, color=INK2, lw=0.8, ls=":"); ax.annotate("euro", (1999.3, 20), fontsize=8, color=INK2); ax.set_title("Real GDP per head, 1995 = 100"); ax.set_ylabel("index"); ax.set_xlim(1960, 2030); save(fig, "16_gdp_per_head_long")
g = np.log(gdppc).diff() * 100; dec = pd.DataFrame({f"{d}s": g.loc[d:d + 9].mean() for d in range(1960, 2020, 10)}); dec["2020-24"] = g.loc[2020:2024].mean()
tab = dec.loc[[ITALY, "DEU", "FRA", "ESP"]]; tab.loc["median of other EU27"] = dec.loc[[c for c in EU27_ISO3 if c != ITALY and c in dec.index]].median()
print("Average annual growth of real GDP per head, % (decades):"); display(tab.round(1))'''),
    (C, '''items = [(long_wide(lh, "gfcf_pct_gdp"), "Investment (GFCF), % of GDP", "%"), (long_wide(lh, "gov_debt_pct_gdp"), "Government debt, % of GDP", "%"),
         (long_wide(lh, "wage_share_adjusted_pct_gdp"), "Adjusted wage share, % of GDP", "%"), (long_wide(lh, "unemployment_rate"), "Unemployment rate", "%"),
         (long_wide(lh, "employment_per_head"), "Persons employed per 100 inhabitants", ""), (long_wide(lh, "gdp_real_per_hour_ncu") / long_wide(lh, "gdp_real_per_hour_ncu").loc[1995] * 100, "Real output per hour, 1995 = 100", "index")]
small_multiples_wide(items, ncols=3, name="16_long_overview")'''),
    (C, '''from cifresignificative.analysis.synth import synthetic_control, placebo_ratios, rmspe
y = np.log(gdppc); y = (y - y.loc[1995]) * 100  # log points relative to 1995
units = [c for c in ["AUT", "BEL", "DNK", "FIN", "FRA", "DEU", "GRC", "IRL", "LUX", "NLD", "PRT", "ESP", "SWE", "GBR", "USA", "JPN", ITALY] if y.loc[1970:2024, c].notna().all()]
donors = [c for c in units if c != ITALY]; pre = (1970, 1994)
w, synth = synthetic_control(y.loc[1970:2024], ITALY, pre, donors)
gap = y[ITALY] - synth
print("Donor weights (> 1%):", {k: round(v, 2) for k, v in w[w > 0.01].sort_values(ascending=False).items()})
print("Pre-1995 fit RMSPE: %.1f log points | gap (Italy minus synthetic) in 2005: %.1f, 2024: %.1f" % (rmspe(y[ITALY], synth, pre), gap.loc[2005], gap.loc[2024]))
ratios = placebo_ratios(y.loc[1970:2024, units], units, pre, (1995, 2024)); print("Post/pre RMSPE ratio: Italy %.1f; rank %d of %d; donors' median %.1f" % (ratios[ITALY], int(ratios.rank(ascending=False)[ITALY]), len(ratios), ratios.drop(ITALY).median()))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(y.loc[1970:2024].index, y.loc[1970:2024, ITALY], color=ITALY_C, lw=2.6, label="Italy"); axes[0].plot(synth.index, synth, color=EU_C, lw=2.2, ls="--", label="synthetic Italy"); axes[0].axvline(1995, color=INK2, lw=0.8, ls=":")
axes[0].set_title("Real GDP per head (log points, 1995 = 0)"); axes[0].legend(fontsize=8)
for u in donors:
    _, s = synthetic_control(y.loc[1970:2024, units], u, pre, [c for c in units if c != u]); axes[1].plot(s.index, y.loc[1970:2024, u] - s, color=PEER_C, lw=0.8, alpha=0.8)
axes[1].plot(gap.index, gap, color=ITALY_C, lw=2.8); axes[1].axhline(0, color=INK2, lw=0.8); axes[1].axvline(1995, color=INK2, lw=0.8, ls=":"); axes[1].set_title("Gap to synthetic control: Italy (blue) and placebo donors (gray)"); axes[1].set_ylabel("log points")
fig.tight_layout(); save(fig, "16_synthetic_control")'''),
]

NOTEBOOKS["17_wages_productivity"] = [
    (M, "# 17 - Pay and productivity: long run, and how much does the deflator matter?\nAMECO long series: nominal compensation per employee deflated by the GDP deflator (a 'producer' real wage), against output per employee and per hour. Then a robustness check of the 2005-2024 real-compensation result in notebook 05 across deflators."),
    (C, '''lh = load_long(); comp = long_wide(lh, "comp_per_employee_nominal_th"); defl = long_wide(lh, "gdp_deflator_2020_100"); gdp = long_wide(lh, "gdp_real_ncu_bn"); emp = long_wide(lh, "employment_th")
real_comp = comp / defl; prod_emp = gdp / emp; prod_hour = long_wide(lh, "gdp_real_per_hour_ncu")
def rebase95(w): return w / w.loc[1995] * 100
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
for ax, c in zip(axes, [ITALY, "DEU", "FRA"], strict=True):
    ax.plot(range(1995, 2025), rebase95(prod_emp).loc[1995:2024, c], color=INK2, lw=2, label="output per employee"); ax.plot(range(1995, 2025), rebase95(real_comp).loc[1995:2024, c], color=ITALY_C if c == ITALY else EU_C, lw=2.4, label="real compensation per employee")
    ax.set_title({"ITA": "Italy", "DEU": "Germany", "FRA": "France"}[c]); ax.set_ylim(80, 160)
axes[0].legend(fontsize=8); axes[0].set_ylabel("1995 = 100"); fig.tight_layout(); save(fig, "17_pay_vs_productivity")
rows = {}
for c in [ITALY, "DEU", "FRA", "ESP", "GBR"]:
    rows[c] = {"output/employee 1995-2024 (%)": (prod_emp.loc[2024, c] / prod_emp.loc[1995, c] - 1) * 100, "real comp 1995-2024 (%)": (real_comp.loc[2024, c] / real_comp.loc[1995, c] - 1) * 100,
               "output/hour 1995-2024 (%)": (prod_hour.loc[2024, c] / prod_hour.loc[1995, c] - 1) * 100, "wage share 1995": long_wide(lh, "wage_share_adjusted_pct_gdp").loc[1995, c], "wage share 2024": long_wide(lh, "wage_share_adjusted_pct_gdp").loc[2024, c]}
display(pd.DataFrame(rows).T.round(1))'''),
    (C, '''# Deflator robustness, 2005-2024: real compensation per employee (2005 = 100) with the HICP vs the GDP deflator
hicp = (1 + wide(panel, "hicp_inflation") / 100).cumprod(); hicp = hicp / hicp.loc[2005]
nom = comp.loc[2005:2024] / comp.loc[2005]
r_hicp = nom / hicp.loc[2005:2024].reindex(columns=nom.columns) * 100; r_gdp = nom / (defl.loc[2005:2024] / defl.loc[2005]) * 100
cmp_ = pd.DataFrame({"HICP": r_hicp.loc[2024], "GDP deflator": r_gdp.loc[2024]}).loc[[c for c in EU27_ISO3 if c in r_hicp.columns]].dropna()
print("Italy 2024 real compensation (2005 = 100): HICP deflator %.1f | GDP deflator %.1f" % (cmp_.loc[ITALY, "HICP"], cmp_.loc[ITALY, "GDP deflator"]))
print("Median of the other members: HICP %.1f | GDP deflator %.1f | Italy's rank among EU27 (1 = lowest): HICP %d, GDP deflator %d" % (cmp_.drop(ITALY).median()["HICP"], cmp_.drop(ITALY).median()["GDP deflator"], cmp_["HICP"].rank()[ITALY], cmp_["GDP deflator"].rank()[ITALY]))
fig, ax = plt.subplots(figsize=(6.4, 5)); scatter_italy(ax, cmp_["HICP"], cmp_["GDP deflator"], "Real compensation 2024, HICP-deflated (2005 = 100)", "GDP-deflator-deflated (2005 = 100)", "Does the choice of deflator change the ranking?"); save(fig, "17_deflator_scatter")
print("rank correlation across countries: %.2f" % cmp_.corr(method="spearman").iloc[0, 1])'''),
    (C, '''# Productivity growth and real pay growth across the EU27, 2005-2019 (average annual log growth, GDP-deflator real pay)
pg = (np.log(prod_hour.loc[2019] / prod_hour.loc[2005]) * 100 / 14); wg = (np.log(real_comp.loc[2019] / real_comp.loc[2005]) * 100 / 14)
fig, ax = plt.subplots(figsize=(6.4, 5)); scatter_italy(ax, pg.reindex(EU27_ISO3), wg.reindex(EU27_ISO3), "Output per hour growth, % per year", "Real compensation per employee growth, % per year", "Pay follows productivity, 2005-2019"); save(fig, "17_pay_productivity_scatter")
d = pd.concat([pg, wg], axis=1).reindex(EU27_ISO3).dropna(); print("corr:", round(d.iloc[:, 0].corr(d.iloc[:, 1]), 2), "| Italy productivity %.2f, real pay %.2f (%% per year); median others %.2f / %.2f" % (d.loc[ITALY].iloc[0], d.loc[ITALY].iloc[1], d.drop(ITALY).iloc[:, 0].median(), d.drop(ITALY).iloc[:, 1].median()))'''),
]

NOTEBOOKS["15_spending_efficiency"].append((M, """## Takeaways (2015-2019 averages, EU27; fits use the other members only)
- **Health looks efficient.** Italy spends 6.8% of GDP in public money on health (close to the median) and has life expectancy 82.9 years (fitted 80.4; +1.0 SD above the relationship; 6th of 27) and treatable mortality of 66.7 per 100,000 (fitted 97.5; 6th best). In 2019 life expectancy was 83.4 (2nd highest; median 81.4), treatable mortality 63.7 (median 80.4) and preventable mortality 101 (median 152; 26th of 27, i.e. among the lowest). Caution: outcomes also reflect diet, climate, age structure and private spending.
- **Education is about as expected, at low spending:** 4.0% of GDP and PISA 481 (average of 2015 and 2018), 6 points above the fitted 475 (+0.3 SD; 12th). Low spending does not translate into the lowest scores, but Italy is not an outlier either way.
- **Old-age spending buys somewhat less poverty protection than the cross-country pattern implies:** 13.4% of GDP, with 15.4% of over-65s below the poverty line against 13.2% predicted (+0.2 SD; 19th). The relationship across countries is weak (poverty also depends on housing, other income and how benefits are targeted).
- **Bottom line for 'bad spending':** these outcome comparisons do not point at health or education; the earlier composition results (notebook 10) point at the size of pensions and interest, and the productivity and procurement evidence (08, 09, 13) at how well public and private resources are used."""))

NOTEBOOKS["16_long_history_counterfactual"].append((M, """## Takeaways (AMECO 1960-2024)
- **Growth faded in steps.** Real GDP per head grew 5.6% a year in the 1960s, 3.2% in the 1970s, 2.2% in the 1980s and 1.4% in the 1990s, then 0.1% in the 2000s and 0.2% in the 2010s (the median of other members: 3.9%, 3.2%, 2.0%, 2.1%, 1.9%, 1.7%). 2020-24 shows a rebound of 1.4% (median 1.3%). Italy was keeping pace until the 1980s and began to lag in the 1990s.
- **Counterfactual.** A synthetic Italy built from other countries (mostly France 42%, Austria 15%, Japan 13%, Finland 9%, USA 9%, Sweden 7%) tracks Italian GDP per head almost perfectly in 1970-1994 (error about 1 log point). From 1995 Italy falls behind: -8 log points by 2005 and -19 by 2024. Italy's post/pre error ratio is 16.4, the largest of the 15 units tested (donors' median 3.1), so the divergence is unusual relative to placebo comparisons.
- **Interpretation limits:** the donors are also exposed to the euro, the single market and global shocks, the pool is small (no donor can isolate one policy), and Germany is excluded because its series start in 1991. Read it as 'Italy's trajectory departs from its previous peer mix after 1995', not as the effect of one event.
- Other long series (investment, debt, unemployment, employment per head, output per hour) are in the figure for context; hours-based productivity has risen only 6% since 1995 vs 33% in Germany (notebook 17)."""))

NOTEBOOKS["17_wages_productivity"].append((M, """## Takeaways (AMECO; 1995-2024 and 2005-2024)
- **Pay tracked a stagnant productivity.** Since 1995 Italian output per hour is up 6.1% (Germany 33.3%, France 26.0%, Spain 18.3%, UK 38.1%); output per employee is down 0.7% and real compensation per employee (GDP-deflated) is down 5.6%, against +14.8% in Germany, +22.2% in France, +4.8% in Spain and +41.1% in the UK. The adjusted wage share eased from 53.8% to 51.1% of GDP (Germany 59.9% to 59.1%).
- **The deflator changes the level, not the ranking.** For 2005-2024, Italian real compensation per employee is 90.5 (2005 = 100) with HICP and 93.8 with the GDP deflator, vs medians of 117.6 and 112.5 for the other members; Italy ranks 2nd or 3rd lowest in the EU27 with either, and the rankings correlate at 0.88. So the finding of a decline in real pay does not hinge on the deflator, though its size does (about -6% to -10%).
- **Pay follows productivity across countries** (correlation 0.85 for 2005-2019 growth rates): Italy sits at the bottom of both, with output per hour +0.03% a year (median 1.01%) and real pay -0.29% a year (median 0.79%).
- Caveat: compensation per employee is a total-economy average and includes composition effects (hours, part-time, sector mix); it is not the wage of a given job."""))

NOTEBOOKS["18_gender_childcare_tax"] = [
    (M, "# 18 - Childcare, women's work and the tax burden by household type\nWho can work, how many hours, and how much of earnings goes in tax and contributions for different households. Eurostat childcare statistics (EU-SILC) and Eurostat's annual net earnings by household case (earn_nt_net)."),
    (C, '''cc = wide(panel, "childcare_under3_30h_plus_pct") + wide(panel, "childcare_under3_1_29h_pct"); fe = wide(panel, "employment_rate_20_64_female"); gap = wide(panel, "employment_rate_20_64_male") - fe
inv = wide(panel, "involuntary_part_time_female_15_64"); W = slice(2022, 2024)
m = lambda w: w.loc[W, [c for c in EU27_ISO3 if c in w]].mean()
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
scatter_italy(axes[0], m(cc), m(fe), "Children under 3 in formal care (%)", "Women's employment rate 20-64 (%)", "Childcare and women's employment")
scatter_italy(axes[1], m(cc), m(gap), "Children under 3 in formal care (%)", "Men minus women employment, pp", "Childcare and the employment gap")
scatter_italy(axes[2], m(wide(panel, "childcare_under3_30h_plus_pct")), m(inv), "Children under 3 in care 30+ hours/week (%)", "Involuntary part-time among part-time women (%)", "Full-time care and involuntary part-time")
fig.tight_layout(); save(fig, "18_childcare_scatters")
x, y = m(cc).dropna(), m(fe)
print("Italy: under-3 formal care %.1f%% (median of others %.1f%%, rank %d/%d); 3-to-school-age early education %.1f%% (median %.1f%%)" % (m(cc)[ITALY], m(cc).drop(ITALY).median(), m(cc).rank(ascending=False)[ITALY], m(cc).notna().sum(), m(wide(panel, "ece_participation_3_to_school_pct"))[ITALY], m(wide(panel, "ece_participation_3_to_school_pct")).drop(ITALY).median()))
d = pd.concat([m(cc), m(fe)], axis=1).dropna(); print("corr(childcare, women's employment) across EU27: %.2f" % d.iloc[:, 0].corr(d.iloc[:, 1]))'''),
    (C, '''e = pd.read_parquet(PROCESSED / "earnings_cases.parquet"); e = e[e.year == 2023]
t = e.pivot_table(index=["iso3", "ecase"], columns="estruct", values="value")
t["tax_ssc_rate"] = (t.TAX + t.SOC) / t.GRS * 100; t["net_rate"] = (t.GRS - t.NET) / t.GRS * 100  # net_rate also counts family benefits
labels = {"P1_NCH_AW50": "Single, 50% of avg wage", "P1_NCH_AW100": "Single, 100%", "P1_NCH_AW167": "Single, 167%", "P1_CH2_AW67": "Single parent, 2 children, 67%", "CPL_CH2_AW100": "Couple, 2 children, 100%"}
eu = [c for c in EU27_ISO3 if c != ITALY]; rows = {}
for case, lab in labels.items():
    s = t.xs(case, level="ecase"); rows[lab] = {"Italy": s.loc[ITALY, "net_rate"], "median others": s.loc[[c for c in eu if c in s.index], "net_rate"].median(), "DEU": s.loc["DEU", "net_rate"], "FRA": s.loc["FRA", "net_rate"], "ESP": s.loc["ESP", "net_rate"],
                                                "Italy tax+SSC rate": s.loc[ITALY, "tax_ssc_rate"], "median tax+SSC": s.loc[[c for c in eu if c in s.index], "tax_ssc_rate"].median(), "Italy family benefit % of gross": s.loc[ITALY, "FAM"] / s.loc[ITALY, "GRS"] * 100}
res = pd.DataFrame(rows).T; display(res.round(1))
fig, ax = plt.subplots(figsize=(8.5, 4)); yy = np.arange(len(res))
ax.barh(yy, res["Italy"], color=ITALY_C, height=0.5, label="Italy", zorder=2); ax.scatter(res["median others"], yy, color=EU_C, marker="D", s=44, zorder=3, label="median of other EU27")
ax.set_yticks(yy, res.index); ax.invert_yaxis(); ax.set_xlabel("% of gross earnings lost to tax and contributions, net of family benefits"); ax.set_title("Tax burden by household type, 2023"); ax.legend(fontsize=8); save(fig, "18_tax_by_household")'''),
]

NOTEBOOKS["19_pension_reform_event_study"] = [
    (M, "# 19 - The 2011 pension reform (Fornero): what happened to older workers?\nA difference-in-differences style event study: Italy minus a control group, relative to the 2008-2010 average, with each other EU country treated in turn as a placebo. Reform in force from January 2012 (law of December 2011): higher retirement ages, faster move to contribution-based pensions. Other countries also reformed during these years and the euro-area crisis hit Italy, so this is suggestive, not a clean causal estimate."),
    (C, '''events = {"employment_rate_55_64": "Employment rate 55-64 (%)", "employment_rate_60_64": "Employment rate 60-64 (%)", "oecd_exit_age_male": "Effective exit age, men (years)"}
reformers = ["ESP", "GRC", "PRT", "FRA", "NLD", "DEU"]  # countries that also raised retirement ages substantially in 2010-2015 (assumption)
BASE = slice(2008, 2010)
def event_paths(ind, treated, controls):
    w = wide(panel, ind); d = w - w.loc[BASE].mean(); return (d[treated] - d[controls].median(axis=1)).loc[2005:2024]
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True); summary = {}
for j, (ind, title) in enumerate(events.items()):
    for i, (lab, excl) in enumerate([("all other EU27", []), ("excluding other reformers", reformers)]):
        ax = axes[i, j]; others = [c for c in EU27_ISO3 if c != ITALY and c not in excl]
        w = wide(panel, ind)
        for u in others:
            if w[u].notna().sum() > 15: ax.plot(event_paths(ind, u, [c for c in others if c != u]), color=PEER_C, lw=0.7, alpha=0.7)
        it = event_paths(ind, ITALY, others); ax.plot(it, color=ITALY_C, lw=2.8); ax.axhline(0, color=INK2, lw=0.8); ax.axvline(2011.5, color=INK2, lw=0.8, ls=":"); ax.set_title(f"{title}\\ncontrols: {lab}", fontsize=9)
        pl = pd.DataFrame({u: event_paths(ind, u, [c for c in others if c != u]) for u in others if w[u].notna().sum() > 15})
        post = pl.loc[2015:2019].mean(); summary[(ind, lab)] = {"Italy 2015-19 gap (pp/years)": it.loc[2015:2019].mean(), "placebo median": post.median(), "Italy rank among countries (1 = largest rise)": int((post > it.loc[2015:2019].mean()).sum()) + 1, "n": len(post) + 1}
fig.tight_layout(); save(fig, "19_pension_reform_event_study"); display(pd.DataFrame(summary).T.round(2))'''),
    (C, '''# Possible spillover on the young: youth unemployment and NEET, same design (confounded by the 2011-13 recession)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (ind, title) in zip(axes, [("youth_unemployment_rate", "Youth unemployment (15-24), pp vs control"), ("neet_rate_15_29", "NEET rate (15-29), pp vs control")], strict=True):
    others = [c for c in EU27_ISO3 if c != ITALY and c not in reformers]; it = event_paths(ind, ITALY, others)
    ax.plot(it, color=ITALY_C, lw=2.6); ax.axhline(0, color=INK2, lw=0.8); ax.axvline(2011.5, color=INK2, lw=0.8, ls=":"); ax.set_title(title)
    print(ind, "gap 2015-19 vs 2008-10 baseline: %.1f pp" % it.loc[2015:2019].mean())
fig.tight_layout(); save(fig, "19_youth_spillover")'''),
]

NOTEBOOKS["20_regional_growth_robustness"] = [
    (M, "# 20 - Regions: productivity vs labour use; and how robust is the growth regression?\nPart 1 splits regional gaps in GDP per head into output per hour (productivity) and hours worked per inhabitant (labour utilisation: employment, hours, demography), for NUTS2 regions across the EU. Part 2 re-runs the cross-country growth regression with different horizons."),
    (C, '''reg = load_regional()
def rw(ind): return reg[reg.indicator == ind].pivot(index="year", columns="geo", values="value")
gdp, hrs, pop = rw("nuts2_gdp_mio_eur"), rw("nuts2_hours_ths"), rw("nuts2_population")
def decomposition(year):
    cols = [c for c in gdp.columns if c in hrs.columns and c in pop.columns]
    d = pd.DataFrame({"y": gdp.loc[year, cols] * 1e6 / pop.loc[year, cols], "p": gdp.loc[year, cols] / hrs.loc[year, cols] * 1e3, "h": hrs.loc[year, cols] * 1e3 / pop.loc[year, cols]}).dropna()
    d["country"] = d.index.str[:2]; return d
Y = 2019; d = decomposition(Y); area = lambda g: {"ITC": "North-West", "ITH": "North-East", "ITI": "Centre", "ITF": "South", "ITG": "Islands"}.get(g[:3], "other")
it = d[d.country == "IT"].assign(area=lambda x: [area(g) for g in x.index]); order = ["North-West", "North-East", "Centre", "South", "Islands"]
tab = it.groupby("area")[["y", "p", "h"]].median().loc[order]; tab.columns = ["GDP per head (EUR)", "GDP per hour (EUR)", "hours per inhabitant"]
display(tab.round(1)); north = it[it.area.isin(["North-West", "North-East"])][["y", "p", "h"]].median(); south = it[it.area.isin(["South", "Islands"])][["y", "p", "h"]].median()
lr = np.log(south / north); print("South & Islands vs North, %d: GDP per head %.0f%% of North; of the log gap %.2f, productivity %.2f (%.0f%%), hours per inhabitant %.2f (%.0f%%)" % (Y, np.exp(lr.y) * 100, lr.y, lr.p, lr.p / lr.y * 100, lr.h, lr.h / lr.y * 100))
rows = {}
for c, g in d.groupby("country"):
    if len(g) < 5: continue
    ly, lp, lh = np.log(g.y), np.log(g.p), np.log(g.h); v = ly.var(); cov = np.cov(lp, lh)[0, 1]
    rows[c] = {"dispersion (sd of log GDP per head)": ly.std(), "share from productivity": (lp.var() + cov) / v, "share from hours per inhabitant": (lh.var() + cov) / v}
dec = pd.DataFrame(rows).T.sort_values("dispersion (sd of log GDP per head)", ascending=False); display(dec.round(2).head(12))
fig, ax = plt.subplots(figsize=(8.5, 4.6)); sel = dec.head(14).iloc[::-1]; yy = np.arange(len(sel))
ax.barh(yy, sel["share from productivity"] * 100, color=[ITALY_C if c == "IT" else PEER_C for c in sel.index], height=0.6, label="productivity (GDP per hour)", edgecolor=SURFACE)
ax.barh(yy, sel["share from hours per inhabitant"] * 100, left=sel["share from productivity"] * 100, color=[EU_C if c == "IT" else "#d8d7d2" for c in sel.index], height=0.6, label="hours worked per inhabitant", edgecolor=SURFACE)
ax.set_yticks(yy, sel.index); ax.set_xlabel("% of regional variation in GDP per head within the country"); ax.set_title(f"Regional inequality: productivity vs labour use, {Y} (Italy highlighted)"); ax.legend(fontsize=8, loc="lower right"); save(fig, "20_regional_decomposition")'''),
    (C, '''# Regional panel with region and year fixed effects: log GDP per head on education, female employment and unemployment (NUTS2, 2013-2023)
from linearmodels.panel import PanelOLS
y = np.log(gdp * 1e6 / pop).stack().rename("ln_gdp_pc"); X = pd.concat({"tertiary": rw("nuts2_tertiary_attainment_25_64").stack(), "female_emp": rw("nuts2_employment_rate_20_64_female").stack(), "unemp": rw("nuts2_unemployment_rate").stack()}, axis=1)
df = pd.concat([y, X], axis=1).dropna().reset_index(); df.columns = ["year", "region", "ln_gdp_pc", "tertiary", "female_emp", "unemp"]; df = df[df.year.between(2013, 2023)]; df["country"] = df.region.str[:2]  # stacking a year x region table gives (year, region)
def fit(sub):
    p = sub.set_index(["region", "year"]); r = PanelOLS(p.ln_gdp_pc, p[["tertiary", "female_emp", "unemp"]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    return pd.DataFrame({"coef": r.params, "p": r.pvalues}), int(r.nobs), r.rsquared_within
out = {}
for lab, sub in [("all EU27 regions", df), ("Italy", df[df.country == "IT"]), ("rest of EU27", df[df.country != "IT"])]:
    t, n, r2 = fit(sub); out[lab] = {"n": n, "within R2": round(r2, 3), **{f"{k} coef": round(t.loc[k, "coef"], 4) for k in t.index}, **{f"{k} p": round(t.loc[k, "p"], 3) for k in t.index}}
display(pd.DataFrame(out).T)'''),
    (C, '''# Growth regression robustness: annual, 3-year and 5-year averages with country and time fixed effects
vars_ = {"gov_debt_pct_gdp": "debt/GDP", "gfcf_pct_gdp": "investment/GDP", "employment_rate_20_64": "employment rate", "tertiary_attainment_25_34": "tertiary 25-34", "old_age_dependency": "old-age dependency"}
dd = panel[panel.iso3.isin(EU27_ISO3) & panel.year.between(2005, 2024)]; wa = dd.pivot_table(index=["iso3", "year"], columns="indicator", values="value")
res = {}
for span in (1, 3, 5):
    recs = []
    for iso, sub in wa.groupby(level=0):
        sub = sub.droplevel(0)
        for start in range(2005, 2025 - span, span):
            win = sub.loc[start + 1:start + span]
            if len(win) == span and win["gdp_growth"].notna().all() and sub.loc[start, list(vars_)].notna().all(): recs.append({"iso3": iso, "window": start, "growth": win["gdp_growth"].mean(), **{vars_[k]: sub.loc[start, k] for k in vars_}})
    p = pd.DataFrame(recs).set_index(["iso3", "window"]); p = p[~p.index.get_level_values("iso3").isin(["IRL"])]  # Ireland's GDP series is dominated by multinational accounting
    r = PanelOLS(p.growth, p[list(vars_.values())], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    res[f"{span}-year windows (n={int(r.nobs)})"] = {**{f"{k}": f"{r.params[k]:+.3f} (p={r.pvalues[k]:.2f})" for k in vars_.values()}, "within R2": round(r.rsquared_within, 3)}
display(pd.DataFrame(res).T)'''),
]

NOTEBOOKS["18_gender_childcare_tax"].append((M, """## Takeaways (Eurostat; 2022-2024 averages for childcare, 2023 for tax)
- **Childcare coverage for under-3s is not an outlier:** 34.9% of children under 3 are in formal care (median of the other members 32.4%; 13th of 27) and 93.4% of children from age 3 to school age attend early education (median 94.0%). Across EU27 countries, coverage is only weakly related to women's employment (correlation 0.19). So coverage at national level does not explain Italy's very low female employment; hours, quality, regional gaps (the South), elder care, job availability and norms matter, and this measure cannot separate them.
- **Tax burden is progressive on single earners:** for a single worker the share of earnings paid in tax and contributions is 12.0% at 50% of the average wage (median 18.2%), 28.3% at 100% (27.4%) and 39.5% at 167% (33.8%). Italy taxes low earners lightly and higher earners heavily relative to peers.
- **Families receive large offsets:** family benefits are worth 13.2% of gross earnings for a couple with two children and 19.8% for a single parent at 67% of the average wage, so the net burden (12.9% and -0.4%) is below the medians (16.2% and 3.7%) even though tax and contributions alone (26.2% for the couple vs 20.7%) are higher.
- Caveat: Eurostat's cases are stylised full-year, full-time workers; they say nothing about part-time work or the informal economy."""))

NOTEBOOKS["19_pension_reform_event_study"].append((M, """## Takeaways (suggestive, not causal)
- **Older-worker employment rose far more in Italy than in controls.** Relative to 2008-2010, employment of 55-64 year-olds in 2015-2019 was 5.8 pp higher than in the control group (placebo median +0.2 pp; Italy 3rd largest rise of about 26 countries) and 60-64 year-olds 8.2 pp (placebo median +0.5; 5th, or 3rd excluding other reforming countries). The effective male exit age rose 0.6 years relative to controls (placebo median -0.1; 7th-8th), a smaller shift than employment.
- **Robust to the choice of controls:** excluding Spain, Greece, Portugal, France, the Netherlands and Germany (which also raised retirement ages in 2010-2015) leaves the 55-64 and 60-64 results essentially unchanged.
- **Youth outcomes cannot be attributed:** Italy's youth unemployment is 13.4 pp and the NEET rate 4.4 pp higher than the controls' relative to 2008-2010, but the 2011-2013 recession hit Italy exactly when the reform took effect, so the event study cannot separate the two (the literature's local-labour-market estimates of a youth-hiring spillover use firm-level variation).
- Caveats: pre-reform trends differ by country (the 2009-2010 reforms already raised retirement ages), the controls' own reforms are not fully excluded, and with 26 countries placebo ranks are a rough guide."""))

NOTEBOOKS["20_regional_growth_robustness"].append((M, """## Takeaways (Eurostat NUTS2, 2019; country panels 2005-2024)
- **Half of the North-South gap is hours, not output per hour.** In 2019 GDP per head in the South and Islands was 56% of the North's; of the log gap, 42% is lower output per hour (EUR 35 vs 45) and 51% fewer hours worked per inhabitant (about 590 vs 790; employment, hours and demography). The same split across the EU's regions: dispersion in Italy (sd of log GDP per head 0.30) is the 7th largest, 47% from productivity and 53% from labour use, against 79% from productivity in Romania and 72% from labour use in Portugal.
- **Regional panel evidence is thin.** With region and year fixed effects, a 1 pp higher female employment rate goes with about 1.3% higher GDP per head across the EU's regions (p<0.001), but none of education, female employment or unemployment is significant in the Italian sample (n=231), and the education and unemployment signs elsewhere (negative and positive) are not interpretable as causal.
- **The growth regression is not robust.** Over 1-, 3- and 5-year horizons with country and time fixed effects and Ireland excluded, no regressor is consistently significant (investment +0.08, -0.04, -0.18; p=0.19, 0.67, 0.15); old-age dependency is significant only in the 5-year windows (p=0.03; n=78 observations, within-R2 not meaningful). This overturns the earlier reading that investment was significant: it depended on including Ireland."""))

NOTEBOOKS["21_crime_migration"] = [
    (M, "# 21 - Crime and migration\nPolice-recorded crime differs across countries in what is counted and how (reporting rates, legal definitions, counting rules), so compare **trends** more than levels; homicide is the most comparable measure. Migration flows are administrative registrations: emigration is under-recorded where people do not deregister (Italy is a known case), so net flows of nationals understate departures. Italy's serious-assault series was dropped because it breaks in 2013-2016."),
    (C, '''items = [
    (wide(panel, "homicide_victims_per_100k"), "Homicide victims per 100,000", "per 100k"), (wide(panel, "robbery_rate_police"), "Police-recorded robbery", "per 100k"), (wide(panel, "theft_rate_police"), "Police-recorded theft", "per 100k"),
    (wide(panel, "burglary_residential_rate_police"), "Burglary of private homes", "per 100k"), (wide(panel, "sexual_violence_rate_police"), "Sexual violence recorded by police", "per 100k"), (wide(panel, "fraud_rate_police"), "Fraud recorded by police", "per 100k"),
    (wide(panel, "prison_population_per_100k"), "Prison population", "per 100k"), (wide(panel, "prison_occupancy_pct"), "Prison occupancy (persons held / capacity)", "%"), (wide(panel, "crime_perception_pct"), "Population reporting crime, violence or vandalism in the area", "% of population"),
]
small_multiples_wide(items, ncols=3, name="21_crime_overview")
fig, ax = plt.subplots(figsize=(7.5, 4)); w = wide(panel, "wb_homicide_rate")
for c, lab in [("USA", "USA"), ("JPN", "Japan"), ("GBR", "UK"), ("DEU", "Germany"), ("FRA", "France")]:
    ax.plot(w.index, w[c], color=PEER_C, lw=1.5); ax.annotate(lab, (w.index[-1], w[c].dropna().iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
ax.plot(w.index, w[ITALY], color=ITALY_C, lw=2.6); ax.annotate("Italy", (w.index[-1], w[ITALY].dropna().iloc[-1]), xytext=(4, 0), textcoords="offset points", color=ITALY_C, fontweight="bold", va="center")
ax.plot(w.index, eu_median_ex_italy(w), color=EU_C, lw=2, ls="--"); ax.set_title("Intentional homicide per 100,000 (UNODC via World Bank)"); ax.set_ylabel("per 100k"); save(fig, "21_homicide_benchmarks")'''),
    (C, '''items = [
    (wide(panel, "immigration_per_1000"), "Immigrants per 1000 inhabitants", "per 1000"), (wide(panel, "emigration_per_1000"), "Emigrants per 1000 (as registered)", "per 1000"), (wide(panel, "net_migration_rate_per_1000"), "Net migration rate incl. adjustment", "per 1000"),
    (wide(panel, "foreign_citizens_share_pct"), "Foreign citizens, % of population", "%"), (wide(panel, "asylum_first_applicants_per_100k"), "First-time asylum applicants per 100,000", "per 100k"), (wide(panel, "first_permits_per_1000"), "First residence permits per 1000 (non-EU)", "per 1000"),
    (wide(panel, "emigration_nationals_per_1000"), "Citizens emigrating per 1000", "per 1000"), (wide(panel, "emigration_nationals_25_34_per_1000"), "Citizens aged 25-34 emigrating, per 1000 of that age", "per 1000"), (wide(panel, "net_migration_nationals_25_34_per_1000"), "Net migration of citizens aged 25-34, per 1000", "per 1000"),
]
small_multiples_wide(items, ncols=3, name="21_migration_overview")'''),
    (C, '''names = ["homicide_victims_per_100k", "robbery_rate_police", "theft_rate_police", "burglary_residential_rate_police", "sexual_violence_rate_police", "fraud_rate_police", "prison_population_per_100k", "prison_occupancy_pct", "crime_perception_pct",
         "immigration_per_1000", "emigration_per_1000", "net_migration_rate_per_1000", "foreign_citizens_share_pct", "asylum_first_applicants_per_100k", "first_permits_per_1000", "emigration_nationals_per_1000", "emigration_nationals_25_34_per_1000", "net_migration_nationals_25_34_per_1000"]
rows = {}
for n in names:
    w = wide(panel, n); y = min(int(w[ITALY].dropna().index.max()), 2024)
    rows[n] = {"year": y, "Italy": round(w.loc[y, ITALY], 2), "median others": round(eu_median_ex_italy(w).loc[y], 2), "rank (1=highest)": "%d/%d" % italy_rank(w, y), "Italy first year": int(w[ITALY].dropna().index.min()), "Italy first value": round(w[ITALY].dropna().iloc[0], 2)}
display(pd.DataFrame(rows).T)'''),
]

NOTEBOOKS["21_crime_migration"].append((M, """## Takeaways (Eurostat; UNODC via World Bank for homicide; 2024 unless stated)
- **Violent crime is low and falling; property crime is high.** Homicide victims fell from 1.05 to 0.57 per 100,000 between 2008 and 2024 (median of the other members 0.89; 23rd of 24 countries, i.e. among the lowest). Police-recorded robbery fell from 110 to 50 per 100,000 but is still twice the median (24; 6th highest); theft is 1,552 vs 727 (7th highest) and burglary of private homes 264 vs 73 (5th of 23; it peaked near 400 in 2012). Recorded fraud has more than doubled since 2016 (300 to 733 vs a median of 417), consistent with online fraud being recorded.
- **Recording differs by country:** recorded sexual violence (12.6 per 100,000 vs a median of 26.6) and drug offences are low partly because of reporting and legal definitions; Italy's serious-assault series breaks between 2012 and 2016 and was dropped.
- **Prisons are crowded:** 108 prisoners per 100,000 (median 114) but occupancy of 122% of official capacity (median 100%; 5th highest; 135% in 2008). The share of people reporting crime or vandalism in their area fell from 14.3% (2005) to 6.4% (2023; median 7.2%).
- **Migration is lower than in most EU countries, except for asylum.** Registered immigration is 7.7 per 1000 inhabitants (median 13.8; 24th of 26), foreign citizens are 8.9% of the population (median 9.2%; 3.9% in 2005), first residence permits 5.9 per 1000 (median 8.1). First-time asylum applicants are 256 per 100,000 (median 87; 9th), up from 51 in 2008.
- **The departure of young Italians is visible in Eurostat data.** Italian citizens aged 25-34 emigrating: 9.0 per 1000 of that age group (median 5.1; 4th highest of 15 countries that report it; 3.8 in 2013). Net migration of citizens aged 25-34 (departures minus returns) is -6.9 per 1000, the lowest of the 15 (median -1.8; -3.1 in 2013). Registered emigration is probably under-counted, so these are lower bounds; the data do not show education, but they are consistent with the brain-drain reports cited in the literature notes (docs/literature)."""))

NOTEBOOKS["22_governments_policies"] = [
    (M, "# 22 - Governments and policy milestones against the data\nItalian governments (dates from Wikipedia, English and Italian editions agree) and 46 laws, decrees and EU/ECB decisions from `data/external/italy_policies.csv`; each entry is labelled *verified* (confirmed in a source during this project), *partly verified* or *unverified* (see the file). Everything below is **descriptive timing, not causal attribution**: policies act with lags, several overlap, and the euro-area environment dominates some series."),
    (C, '''from cifresignificative.analysis.timeline import load_governments, load_policies, government_stats
gov = load_governments("2001-01-01"); pol = load_policies(); lh = load_long()
print(len(pol), "policies:", pol.evidence.value_counts().to_dict())
stats = government_stats(lh, load_governments("1989-01-01"), "1989-01-01")
st = stats.dropna(subset=["gdp_growth_pa_italy", "gdp_growth_pa_others"]).copy()
slower = (st.gdp_growth_pa_italy < st.gdp_growth_pa_others).sum()
print(f"Italy grew more slowly than the median of the other members under {slower} of {len(st)} governments (weighted by time in office); faster under:", st[st.gdp_growth_pa_italy >= st.gdp_growth_pa_others].government.tolist())
show = stats[["government", "start", "months", "gdp_growth_pa_italy", "gdp_growth_pa_others", "unemployment_rate_italy", "unemployment_rate_others", "debt_change_pp_italy", "debt_change_pp_others"]].round(1)
display(show.set_index("government"))'''),
    (C, '''fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
s = stats[stats.start >= pd.Timestamp("1994-01-01").date()].reset_index(drop=True); yy = np.arange(len(s))
axes[0].barh(yy - 0.18, s.gdp_growth_pa_italy, height=0.34, color=ITALY_C, label="Italy"); axes[0].barh(yy + 0.18, s.gdp_growth_pa_others, height=0.34, color=EU_C, label="median of other EU27")
axes[0].set_yticks(yy, [f"{g} ({st.year})" for g, st in zip(s.government, s.start, strict=True)]); axes[0].invert_yaxis(); axes[0].axvline(0, color=INK2, lw=0.8); axes[0].set_title("Real GDP growth, % per year while in office"); axes[0].legend(fontsize=8)
d = s.dropna(subset=["debt_change_pp_italy"]); axes[1].barh(np.arange(len(d)) - 0.18 + (len(s) - len(d)) * 0, d.debt_change_pp_italy, height=0.34, color=ITALY_C); axes[1].barh(np.arange(len(d)) + 0.18, d.debt_change_pp_others, height=0.34, color=EU_C)
axes[1].axvline(0, color=INK2, lw=0.8); axes[1].set_title("Change in debt/GDP during the term, pp"); fig.tight_layout(); save(fig, "22_government_stats")'''),
    (C, '''def panel_for(ax, series, title, ylabel, policies, xlim=(2005, 2025), median=None):
    ax.plot(series.index, series.values, color=ITALY_C, lw=2.4)
    if median is not None: ax.plot(median.index, median.values, color=EU_C, lw=1.8, ls="--")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlim(xlim[0], xlim[1] + 1)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.08 * (hi - lo), hi)
    draw_timeline(ax, gov, policies, xlim=xlim)
sel = lambda pat: pol[pol.domain.str.contains(pat, case=False)]
fig, axes = plt.subplots(3, 2, figsize=(13, 12))
w = wide(panel, "employment_rate_55_64"); panel_for(axes[0, 0], w[ITALY], "Employment rate 55-64 and pension reforms", "%", sel("pension"), median=eu_median_ex_italy(w))
w = wide(panel, "youth_unemployment_rate"); panel_for(axes[0, 1], w[ITALY], "Youth unemployment and labour reforms", "%", sel("labour"), median=eu_median_ex_italy(w))
w = wide(panel, "vat_compliance_gap_pct"); panel_for(axes[1, 0], w[ITALY].dropna(), "VAT compliance gap and tax-compliance measures", "% of VAT due", sel("tax"), median=eu_median_ex_italy(w).dropna())
w = wide(panel, "asylum_first_applicants_per_100k"); panel_for(axes[1, 1], w[ITALY].dropna(), "First-time asylum applicants and immigration laws", "per 100k", sel("immigration"), median=eu_median_ex_italy(w).dropna())
y = wide(panel, "long_term_yield"); spread = (y[ITALY] - y["DEU"]).dropna(); panel_for(axes[2, 0], spread, "10-year yield spread vs Germany and ECB decisions", "pp", sel("monetary|EU / inv|euro"))
sp = load_spending(); cap = spend_wide(sp, "D9PAY", "nature"); panel_for(axes[2, 1], cap[ITALY].dropna(), "Capital transfers and the Superbonus", "% of GDP", pol[pol.title.str.contains("Superbonus|Superbonus credit", case=False)], median=eu_median_ex_italy(cap).dropna())
fig.suptitle("Blue = Italy, orange dashed = median of other EU27, diamonds = policy dates (filled = verified), shaded bands = governments", fontsize=9, color=INK2, x=0.01, ha="left")
fig.tight_layout(); save(fig, "22_timelines")'''),
    (C, '''# The policy file itself, for reference (verified entries first)
cols = ["date", "title", "domain", "reference", "evidence"]
display(pol.sort_values(["evidence", "date"], key=lambda s: s.map({"verified": 0, "partly verified": 1, "unverified": 2}) if s.name == "evidence" else s)[cols].reset_index(drop=True))'''),
]

NOTEBOOKS["22_governments_policies"].append((M, """## Takeaways (descriptive timing; policy dates labelled by confidence in `data/external/italy_policies.csv`)
- **Growth has been below the median of the other members under almost every government.** Weighted by time in office, Italy grew more slowly than the median of the other EU27 members under 21 of 22 governments since 1989 (D'Alema II, four months, is counted with D'Alema I, the same prime minister and coalition; De Mita counts for his whole term from April 1988); the exception is Draghi (7.0% a year vs 5.2%), a rebound after the pandemic fall under Conte II (-5.3% vs -1.2%). The gap was widest under Berlusconi II (0.8% vs 3.3%, 46 months), Prodi II (1.1% vs 4.1%), Renzi (0.7% vs 2.8%) and Conte I (0.6% vs 2.9%); it was negative growth under Berlusconi IV (-1.1% vs +0.2%) and Monti (-2.5% vs +0.4%). The shortfall therefore spans governments of all colours, which points to structural causes rather than to any one coalition (the euro-area crisis explains part of 2008-2013).
- **Debt moves are dominated by crises and by the pandemic.** Debt/GDP fell by 13.6 pp under Draghi (median of others -4.2 pp, helped by high inflation and growth), rose 19.4 pp under Conte II (others +10.2) and 14.8 pp under Berlusconi IV (others +16.1); it fell under Prodi I (-4.6 pp vs -1.1) and Berlusconi II (-3.5 vs 0.0).
- **Unemployment gaps widened after 2011:** Italy's unemployment during the Monti, Letta, Renzi, Gentiloni and Conte I governments was 11.0%, 12.5%, 12.2%, 11.1% and 10.2% against medians of 9.8%, 9.7%, 8.7%, 6.6% and 5.6%.
- **Timelines.** (1) Employment of 55-64 year-olds rose from 31.4% (2005) to 36.5% (2010), 48.2% (2015), 54.3% (2019) and 59.0% (2024) through the Maroni, Damiano, Sacconi and Fornero measures and Quota 100; it crossed the median around 2015-2017 and fell behind again after 2019 as the median kept rising. (2) The spread against Germany peaked at 4.0 pp in 2012 and fell to 1.2 pp by 2015, between the ECB's July and September 2012 announcements and the start of QE in March 2015. (3) The VAT gap fell from 24.4% (2015) to 20.9% (2017) and 19.3% (2019), then to 15.1% (2021): the decline started before electronic invoicing became mandatory (January 2019) and continued after it, so the data cannot isolate its effect. (4) First-time asylum applicants per 100,000 peaked at 211 in 2017, fell to 89 in 2018 and 59 in 2019 (Minniti-Orlando decrees, Salvini decree), then rose to 256 in 2024. (5) Capital transfers went from 1.0% of GDP (2019) to 5.0% (2021) and 5.9% (2023), then 1.7% (2024), bracketing the Superbonus (May 2020) and the credit-transfer block (February 2023).
- **Limits:** no counterfactual, many overlapping measures, lagged effects; the euro-area environment (2008-2012 crisis, 2012 ECB announcements, 2015 QE, 2020 PEPP) explains much of the timing in debt, yields and growth. Three policy dates are still unconfirmed in the file (80-euro bonus, bail-in application, Spazzacorrotti law) and seven are partly confirmed."""))

NOTEBOOKS["12_higher_education"].append((M, "## Adult skills (PIAAC, 2023 and change since 2012)\nOECD Survey of Adult Skills, adults aged 16-65. 19 EU members took part (Belgium and the UK only as regions, so they are absent). 2012 levels are the 2023 level minus the OECD's reported change. Scores are on a 0-500 scale."))
NOTEBOOKS["12_higher_education"].append((C, '''from cifresignificative.countries import NAMES
keys = ["literacy", "numeracy", "adaptive_problem_solving", "low_literacy_pct", "literacy_16_24", "literacy_25_34", "literacy_55_65", "literacy_women", "literacy_men",
        "literacy_below_upper_secondary", "literacy_tertiary", "literacy_parents_no_upper_secondary", "literacy_parent_tertiary"]
pl = {k: wide(panel, f"piaac_{k}") for k in keys}
y23 = pl["literacy"].loc[2023, [c for c in pl["literacy"] if c in EU27_ISO3 or c in BENCHMARKS]].dropna().sort_values()
y12 = pl["literacy"].loc[2012].reindex(y23.index)
fig, ax = plt.subplots(figsize=(7.5, 5.2)); yy = np.arange(len(y23))
for i, c in enumerate(y23.index):
    if not np.isnan(y12[c]): ax.plot([y12[c], y23[c]], [i, i], color=PEER_C if c != ITALY else ITALY_C, lw=1.4)
ax.scatter(y12.values, yy, s=22, facecolor="none", edgecolor=INK2, zorder=3, label="2012")
ax.scatter(y23.values, yy, s=[70 if c == ITALY else 36 for c in y23.index], color=[ITALY_C if c == ITALY else PEER_C for c in y23.index], edgecolor=SURFACE, zorder=4, label="2023")
ax.set_yticks(yy, [NAMES.get(c, c) for c in y23.index]); ax.axvline(pl["literacy"].loc[2023].drop(ITALY).median(), color=EU_C, ls="--", lw=1.4)
ax.set_xlabel("Mean literacy score"); ax.set_title("Adult literacy, 2012 (hollow) and 2023"); ax.legend(fontsize=8, loc="lower right"); save(fig, "12_piaac_literacy")
def row_of(w): return {"Italy": w.loc[2023, ITALY], "median others": eu_median_ex_italy(w, 0.5).loc[2023], "rank (1=highest)": "%d/%d" % italy_rank(w, 2023)}
tab = pd.DataFrame({"literacy": row_of(pl["literacy"]), "numeracy": row_of(wide(panel, "piaac_numeracy")), "adaptive problem solving": row_of(pl["adaptive_problem_solving"]), "% at level 1 or below (literacy)": row_of(pl["low_literacy_pct"])}).T
display(tab.round(1))
groups = {"16-24": "literacy_16_24", "25-34": "literacy_25_34", "55-65": "literacy_55_65", "women": "literacy_women", "men": "literacy_men", "below upper secondary": "literacy_below_upper_secondary", "tertiary": "literacy_tertiary",
          "parents: none upper secondary": "literacy_parents_no_upper_secondary", "parent tertiary": "literacy_parent_tertiary"}
g = pd.DataFrame({lab: {"Italy": pl[k].loc[2023, ITALY], "median others": eu_median_ex_italy(pl[k], 0.5).loc[2023], "gap": pl[k].loc[2023, ITALY] - eu_median_ex_italy(pl[k], 0.5).loc[2023]} for lab, k in groups.items()}).T
display(g.round(1))
mob = (pl["literacy_parent_tertiary"] - pl["literacy_parents_no_upper_secondary"]).loc[2023].dropna()
print("Literacy gap between children of tertiary-educated and low-educated parents: Italy %.0f points vs median of others %.0f (rank %d/%d, 1 = largest gap)" % (mob[ITALY], mob.drop(ITALY).median(), mob.rank(ascending=False)[ITALY], len(mob)))
sk = (pl["literacy_tertiary"] - pl["literacy_below_upper_secondary"]).loc[2023].dropna()
print("Tertiary vs below-upper-secondary literacy gap: Italy %.0f vs median %.0f" % (sk[ITALY], sk.drop(ITALY).median()))
chg = (pl["literacy"].loc[2023] - pl["literacy"].loc[2012]).dropna(); print("Change in literacy 2012-2023: Italy %+.0f | median of others %+.0f | countries with data %d" % (chg[ITALY], chg.drop(ITALY).median(), len(chg)))'''))

NOTEBOOKS["12_higher_education"].append((M, """## Takeaways: adult skills (PIAAC, OECD Education GPS export; 19 EU members, 2023)
- **Italy scores below most participating countries.** Mean literacy 245 (median of the other EU members 254.5; 16th of 19), numeracy 244 (262; 17th) and adaptive problem solving 231 (248.5; 17th). 34.7% of adults are at level 1 or below in literacy (median 26.2%; 4th highest of 19; Germany 22.5%).
- **Not a recent decline relative to peers.** Mean literacy fell by 5 points from 250 in 2012 to 245 in 2023, while the median of the other countries with data fell by 6. The gap is long-standing rather than new.
- **Young Italians catch up only partly.** Scores are 263 for 16-24 year-olds (median 272.5), 253 for 25-34 (268.5; the widest gap, -15.5 points) and 232 for 55-65 (240.5). Younger cohorts score higher than older ones, as elsewhere, but the gap to peers does not close.
- **Graduates are where the shortfall is.** Adults below upper-secondary education score *above* peers (217 vs 207) but tertiary graduates score 271 vs 283.5. A degree adds 54 literacy points over the lowest education level in Italy, against 72 in the median country: low attainment and weaker skills of graduates reinforce each other.
- **Family background matters less than elsewhere, from a lower starting point.** Children of tertiary-educated parents score 272 (median 283); of parents without upper-secondary schooling 233 (median 232.5): the gap by parental education is 39 points against 48 in the median country (14th of 21).
- **Gender:** women 247, men 244, a small female advantage like the median. Caveats: Belgium and the UK are only partly covered, 2012 levels are derived from reported changes, and the 2023 survey's cross-country comparability is stated by the OECD only within its published confidence intervals (not reproduced here)."""))

NOTEBOOKS["23_growth_accounting"] = [
    (M, "# 23 - Growth accounting by industry (EU KLEMS & INTANProd)\nLabour productivity per hour growth split into contributions of total factor productivity (TFP), labour composition (skills, experience), ICT capital, other tangible capital and intangible capital, for the market economy and by industry, 1996-2019 (EU KLEMS release 2025). Contributions are in percentage points per year. Notebooks 08 and 13 showed *where* Italy's productivity lags; this shows *what kind* of growth is missing."),
    (C, '''ek = pd.read_parquet(PROCESSED / "euklems_growth.parquet")
def avg(nace, a, b):
    d = ek[(ek.nace_r2_code == nace) & ek.year.between(a, b)]; return d.pivot_table(index="iso3", columns="variable", values="value", aggfunc="mean")
parts = {"LP1ConTFP": "TFP", "LP1ConLC": "labour composition", "LP1ConTangICT": "ICT capital", "LP1ConTangNICT": "other tangible capital", "LP1ConIntang": "intangible capital"}
cols = ["#2a78d6", "#eda100", "#1baf7a", "#8a8a85", "#e87ba4"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True); summ = {}
for ax, (a, b) in zip(axes, [(1996, 2019), (1996, 2007), (2008, 2019)], strict=True):
    t = avg("MARKT", a, b); oth = t.drop(index=ITALY).loc[[c for c in t.index if c in EU27_ISO3 and c != ITALY]]
    sel = t.loc[[ITALY, "DEU", "FRA", "ESP"]]; sel.loc["median of other EU27"] = oth.median(); sel["LP1_G"] = sel["LP1_G"]
    pos = np.zeros(len(sel)); neg = np.zeros(len(sel))
    for key, col in zip(parts, cols, strict=True):
        v = sel[key].values; base = np.where(v >= 0, pos, neg); ax.bar(range(len(sel)), v, bottom=base, color=col, width=0.62, edgecolor=SURFACE, label=parts[key]); pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
    ax.scatter(range(len(sel)), sel["LP1_G"], color=INK, s=26, zorder=5, label="labour productivity growth"); ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(sel)), ["Italy", "Germany", "France", "Spain", "median\\nothers"], fontsize=8); ax.set_title(f"{a}-{b}"); summ[f"{a}-{b}"] = sel.round(2)
axes[0].set_ylabel("pp per year, market economy"); axes[0].legend(fontsize=7, loc="lower left"); fig.suptitle("What drove labour productivity growth per hour (EU KLEMS)", x=0.01, ha="left", fontsize=11); fig.tight_layout(); save(fig, "23_growth_accounting")
for k, v in summ.items(): print(k); display(v)
print("countries with market-economy accounts:", avg("MARKT", 1996, 2019).LP1_G.notna().sum())'''),
    (C, '''# TFP contribution by industry, 1996-2019: Italy vs the median of the other EU members
secs = {"A": "Agriculture", "B": "Mining", "C": "Manufacturing", "D-E": "Utilities & waste", "F": "Construction", "G": "Trade", "H": "Transport & storage", "I": "Accommodation & food", "J": "Information & communication", "K": "Finance", "M-N": "Professional & admin services", "R-S": "Arts & other services"}
rows = {}
for n, lab in secs.items():
    t = avg(n, 1996, 2019)
    if ITALY in t.index and "LP1ConTFP" in t:
        oth = t.loc[[c for c in t.index if c in EU27_ISO3 and c != ITALY], "LP1ConTFP"].dropna(); rows[lab] = {"Italy": t.loc[ITALY, "LP1ConTFP"], "median others": oth.median(), "n": len(oth), "Italy labour productivity growth": t.loc[ITALY, "LP1_G"], "median LP growth": t.loc[oth.index, "LP1_G"].median()}
tab = pd.DataFrame(rows).T.sort_values("Italy"); display(tab.round(2))
fig, ax = plt.subplots(figsize=(8, 4.4)); yy = np.arange(len(tab))
ax.barh(yy, tab["Italy"], color=[ITALY_C if v >= 0 else EU_C for v in tab["Italy"]], height=0.5, label="Italy"); ax.scatter(tab["median others"], yy, color=INK2, marker="D", s=38, zorder=3, label="median of other EU members")
ax.set_yticks(yy, tab.index); ax.axvline(0, color=INK2, lw=0.8); ax.set_xlabel("TFP contribution to labour productivity growth, pp per year"); ax.set_title("TFP growth by industry, 1996-2019"); ax.legend(fontsize=8, loc="lower right"); save(fig, "23_tfp_by_industry")'''),
    (C, '''# TFP level index (1995 = 100), market economy
t = ek[(ek.nace_r2_code == "MARKT") & (ek.variable == "LP1TFP_I")].pivot(index="year", columns="iso3", values="value"); t = t / t.loc[1995] * 100
fig, ax = plt.subplots(figsize=(7.5, 4)); oth = t[[c for c in t if c in EU27_ISO3 and c != ITALY]]
for c in oth: ax.plot(oth.index, oth[c], color=PEER_C, lw=0.7, alpha=0.7)
ax.plot(oth.index, oth.median(axis=1), color=EU_C, lw=2, ls="--"); ax.plot(t.index, t[ITALY], color=ITALY_C, lw=2.6); ax.set_ylim(60, 140); ax.set_title("TFP index, market economy (1995 = 100)"); save(fig, "23_tfp_index")
print("TFP 1995-2019: Italy %.0f | DEU %.0f | FRA %.0f | ESP %.0f | median of others %.0f" % (t.loc[2019, ITALY], t.loc[2019, "DEU"], t.loc[2019, "FRA"], t.loc[2019, "ESP"], oth.loc[2019].median()))'''),
]

NOTEBOOKS["23_growth_accounting"].append((M, """## Takeaways (EU KLEMS & INTANProd, release 2025; market economy; pp per year unless stated)
- **The shortfall is TFP, not capital or skills.** Labour productivity per hour grew 0.60% a year in 1996-2019 in Italy (Germany 1.45%, France 1.32%, Spain 0.57%; the median of the other members is higher because it includes fast-converging eastern economies: 2.35%). Italy's contribution from total factor productivity was -0.36 (Germany +0.85, France +0.19, Spain -0.54, median of the others +0.79), while labour composition (+0.36, median +0.36), ICT capital (+0.04 vs +0.02), other tangible capital (+0.30 vs +0.22) and intangible capital (+0.25 vs +0.35) are close to the median. TFP in the market economy is 8% *below* its 1995 level in 2019 (Germany +23%, France +5%, Spain -12%).
- **Two phases.** Before 2008 Italy's TFP contribution was -0.22 a year while Germany's was +1.42, France's +0.97 and the median's +1.37; after 2008 it was -0.49 (Germany +0.28, France -0.58, Spain -0.33, median +0.21). So Italy's weakness predates the crisis, while after 2008 France and Spain also had negative TFP contributions and the gap to the median narrowed.
- **By industry (1996-2019, 13-15 countries with industry accounts).** TFP contributions are most negative in utilities and waste (-3.8 vs a median of -1.1), construction (-2.0 vs -0.2), professional and administrative services (-1.3 vs -0.1). The largest gap is in manufacturing: -0.43 in Italy against +1.55 for the median (about 2 points a year), followed by agriculture (+0.37 vs +1.63) and information and communication (+0.34 vs +1.15). Finance (+0.85 vs +0.28) and mining are exceptions. This complements notebook 13: both show weakness in construction and professional services, while the manufacturing gap is clearer in TFP terms than in output per hour.
- **Caveats:** TFP is a residual that absorbs mismeasurement (capital utilisation, services quality, firm composition); industry accounts exist for fewer countries than the market-economy aggregate; the 'median of the other members' includes catching-up economies. The data cannot say whether the TFP shortfall reflects firm size, regulation or allocation (see the literature notes: Hsieh-Klenow 2009; Gopinath et al. 2017), which needs firm-level data."""))


NOTEBOOKS["24_correlation_discovery"] = [
    (M, """# Correlation discovery
Which indicators move together across the EU27 + GBR/USA/JPN, and where does Italy sit relative to those relationships?
**Associations only, never causal.** About 30 countries and ~15,000 indicator pairs means a naive screen is mostly noise,
so every screen below is followed by Benjamini-Hochberg false-discovery control and robustness stamps
(leave-one-country-out, without Ireland, Spearman vs Pearson). Size-scaled series (counts, EUR millions) and near-duplicates
(|rho| >= 0.95 between differently named series) are dropped. Code: `src/cifresignificative/analysis/corr.py`, `experiments.py`."""),
    (C, '''from cifresignificative.analysis import corr, experiments as ex
M_ = corr.panel_matrix(panel)
res = ex.run_all(panel)
funnel = pd.DataFrame({"survivors (FDR 5%, not duplicate)": {k: len(v) for k, v in res.items() if k.startswith(("e1", "e2", "e3"))},
                       "robust": {k: int(v["robust"].sum()) if "robust" in v else np.nan for k, v in res.items() if k.startswith(("e1", "e2", "e3"))}})
display(funnel)'''),
    (M, "## Why we control for income\nRich countries are better on almost everything, so a raw cross-section mostly rediscovers one 'development' factor. `e1_levels` correlates what is left after removing the log-GDP-per-head fit; `e1_levels_raw` does not."),
    (C, '''top = res["e1_levels"].head(15)[["x", "y", "n", "rho", "q_value", "robust"]]
display(top.round(3))
# null check: shuffling each indicator across countries should leave ~0 discoveries
X1 = corr.cross_section(M_, "level", (2015, 2019))
print("discoveries on shuffled data:", corr.permutation_null(X1, n_perm=3))'''),
    (M, "## E2 - do improvements co-occur? (change 2005-09 to 2015-19)"),
    (C, 'display(res["e2_changes"].head(15)[["x", "y", "n", "rho", "q_value", "robust"]].round(3))'),
    (M, "## E3 - within-country co-movement (year-on-year changes, demeaned by country)"),
    (C, 'display(res["e3_within"].head(15)[["x", "y", "n", "rho", "q_value"]].round(3))'),
    (M, "## E5 - what moves with Italy's gap?\nSpearman correlation of year-on-year changes in the gap (Italy minus median of the other EU27) for each indicator against the gap in the target."),
    (C, '''for t in ("e5_italy_gap_gdp", "e5_italy_gap_employment"):
    print(t); display(res[t].head(8).round(3))'''),
    (M, "## E7 - literature hypotheses with panel fixed effects\nEntity + year FE, entity-clustered SEs, also without Ireland. `italy_resid_z` = Italy's distance from the cross-country line fitted on the other countries (2015-19), in residual SDs."),
    (C, '''h = res["e7_hypotheses"]
display(h[["id", "label", "expected_sign", "coef", "p", "q_value", "sign_agrees", "p_ex_irl", "italy_resid_z", "n_obs"]].round(3))'''),
    (M, "## E4 - lead-lag\nFor each hypothesis: does x(t-k) predict y(t) (entity + year FE), and does y(t-k) predict x(t)? Verdict 'x leads' / 'y leads' when only one direction is significant at 5%. Predictive order only."),
    (C, '''ll = res["e4_lead_lag"]
display(ll.pivot_table(index=["id", "x", "y"], columns="lag", values="verdict", aggfunc="first"))'''),
    (C, '''fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (x, y) in zip(axes, [("pension_spend_pct_gdp", "neet_rate_15_29"), ("early_leavers_18_24", "neet_rate_15_29"), ("gdp_growth", "gov_debt_pct_gdp")]):
    d = X1[[x, y]].dropna()
    ax.scatter(d[x], d[y], color="0.6")
    ax.scatter(d.loc[["ITA"], x], d.loc["ITA", y], color="C0", zorder=3)
    b, a = np.polyfit(d.drop("ITA")[x], d.drop("ITA")[y], 1)
    xs = np.linspace(d[x].min(), d[x].max(), 20); ax.plot(xs, a + b * xs, "--", color="C1")
    ax.set_xlabel(x); ax.set_ylabel(y)
plt.tight_layout(); plt.show()'''),
]

NOTEBOOKS["24_correlation_discovery"].append((M, """## Takeaways (Eurostat, World Bank, IMF, AMECO, OECD; descriptive associations only)
- **Income drives most of the raw signal.** The 2015-19 cross-section yields about 1,000 FDR-significant pairs; after removing the log-GDP-per-head fit about 200 remain. Shuffling indicators across countries gives 0 discoveries, so the screen is not producing noise on its own.
- **The leaders are mostly definitional or same-family pairs** (life expectancy vs preventable mortality, PISA maths vs science, unemployment vs youth unemployment). Name-based filters catch the obvious ones; the list still needs human curation before any pair is read as a finding. Robustness checks barely discriminate among FDR survivors (nearly all pass), so they flag fragile pairs rather than rank good ones.
- **Italy's gap in GDP per head moves with its gaps in debt and pensions.** Year-on-year changes in the gap correlate at -0.71 with government debt/GDP, -0.71 with the IMF debt series (the same quantity, so not independent) and -0.68 with pension spending (q = 0.03 to 0.05, n = 19-20 years). No indicator passes FDR for the employment-rate gap (best: current account, rho = -0.62, q = 0.19).
- **Literature hypotheses (panel FE, entity + year effects):** 9 of 13 are significant at q < 0.05, and 7 of those go the expected way (TFP-GDP, yields-investment, growth-debt, pensions-55-64 employment, pensions-NEET, female employment-GDP, early leaving-NEET). Two significant results go the *opposite* way to the stated expectation (H4 public investment vs balance, H5 pensions vs education spending) and should be treated as refuted or mis-specified, not as support. Four show no significant association (graduates-emigration, childcare-female employment, TFP-net earnings, government effectiveness-GDP; q > 0.2).
- **Italy against the cross-country line (2015-19):** NEET is 2.6 residual SDs above the line implied by pension spending and 2.6 above the one implied by early school leaving; female employment is 2.3 SDs below the line implied by full-time childcare coverage.
- **Lead-lag (predictive order, 1-3 years):** female employment predicts GDP per head at every lag but not the reverse (H9); long-term yields predict investment (H2); early school leaving predicts NEET at 2-3 years (H13); pension spending predicts 55-64 employment at 2-3 years (H7). The reverse direction dominates for childcare vs female employment (H10), budget balance vs public investment (H4) and pension vs education spending (H5), so the stated direction of those hypotheses is not supported. TFP-GDP (H1) is significant both ways, and TFP-earnings, government effectiveness-GDP and graduates-emigration show nothing either way.
- **Limits:** ~30 countries and 15-20 years; provisional and broken observations are dropped; same-quantity pairs under different names can survive; no causal identification, and lead-lag tests only show predictive order."""))


NOTEBOOKS["25_regional_correlation_discovery"] = [
    (M, """# Regional correlation discovery (NUTS2)
The national screen (notebook 24) has about 30 countries. Here the unit is the **region** (about 230 NUTS2 regions with data in the 2015-19 window, 26 indicators plus derived ratios), which gives far more observations but needs care:
regions of one country share institutions, policy and the national cycle, and neighbours are not independent. So:
- the default screen works **within countries** (each indicator minus its country mean; countries with fewer than 3 regions drop out) and removes the log-GDP-per-head fit;
- robustness drops **whole countries** (leave-one-country-out); the permutation null **shuffles regions inside each country**;
- derived ratios that share a numerator or denominator (output per hour, pay per hour, labour share, hours per inhabitant, investment rate) are not paired with each other;
- Italy is tested with a **placebo ranking** (the same slope difference for every country with at least 5 regions) because Italy is a single cluster.
**Associations only, never causal.** Code: `src/cifresignificative/analysis/regional_corr.py`, `experiments.py`."""),
    (C, '''from cifresignificative.analysis import corr, regional_corr as rc, experiments as ex
reg = load_regional()
res = ex.run_regional_all(reg)
funnel = pd.DataFrame({"survivors": {k: len(v) for k, v in res.items() if k.startswith(("r1", "r2"))},
                       "robust": {k: int(v["robust"].sum()) for k, v in res.items() if k.startswith(("r1", "r2"))}})
display(funnel)
R = rc.regional_matrix(reg, drop_flag_chars="u")
X1 = rc.within_country_table(R, "level", (2015, 2019))
print("regions with data per indicator (2015-19):"); display(X1.notna().sum().sort_values().to_frame("regions").T)
print("discoveries after shuffling regions within countries:", rc.region_permutation_null(X1, n_min=80, n_perm=3))'''),
    (M, "## R1 - within-country associations, 2015-19 (income removed)\nSame-theme pairs (labour vs labour, demography vs demography) are near-definitional; the table keeps robust pairs from different themes."),
    (C, '''theme_of = reg.assign(indicator=reg.indicator.str.removeprefix("nuts2_")).groupby("indicator").theme.first()
r1 = res["r1_levels_within"]
cross = r1[r1.robust & (r1.x.map(theme_of) != r1.y.map(theme_of))].assign(x_theme=lambda d: d.x.map(theme_of), y_theme=lambda d: d.y.map(theme_of))
display(cross.head(20)[["x", "x_theme", "y", "y_theme", "n", "rho", "q_value"]].round(3))
print("between countries (n = countries):"); display(res["r1_levels_between"].head(8)[["x", "y", "n", "rho", "q_value", "robust"]].round(3))'''),
    (M, "## R2 - within-country associations in changes (2005-09 to 2015-19; break-flagged and provisional values dropped)"),
    (C, 'display(res["r2_changes_within"].head(15)[["x", "y", "n", "rho", "q_value", "robust"]].round(3))'),
    (M, "## R2b - permutation check of the p-values behind R1 and R2\nThe pooled Spearman p assumes countries share one dispersion; they do not, and a shuffled test rejects more than the nominal 5%. Here every pair of R1 and R2 is retested with the within-country permutation p (`regional_south.perm_p`, 20,000 shuffles, ranks pooled), BH over all pairs. The funnel's duplicate and robustness steps are not repeated."),
    (C, '''from cifresignificative.analysis import regional_south as rs
chk = pd.DataFrame({k: {"pairs tested": len(res[k]), "BH survivors, asymptotic p": int(res[k].significant_asymptotic.sum()), "BH survivors, permutation p": int(res[k].significant.sum()),
                        "both": int((res[k].significant & res[k].significant_asymptotic).sum()), "share p < 0.05, asymptotic": round(float((res[k].p_asymptotic < 0.05).mean()), 3), "share p < 0.05, permutation": round(float((res[k].p < 0.05).mean()), 3)}
                    for k in ["rp1_levels_perm", "rp2_changes_perm"]})
display(chk)
lost = pd.concat([res[k][res[k].significant_asymptotic & ~res[k].significant].assign(screen=k[:3]) for k in ["rp1_levels_perm", "rp2_changes_perm"]])
print("asymptotic survivors that lose significance under the permutation p:"); display(lost[["screen", "x", "y", "n", "rho", "p_asymptotic", "p", "q_value"]].round(4))
rng = np.random.default_rng(0); g = rc.country_of(X1.index); Xn = X1.drop(columns=corr.size_scaled(X1.columns)); null = []
for k in range(50):
    t = rs.screen_perm(corr.shuffle_within(Xn, g, rng), n_perm=1000, seed=k)
    null.append({"discoveries, asymptotic": int(t.significant_asymptotic.sum()), "discoveries, permutation": int(t.significant.sum()), "share p < 0.05, asymptotic": (t.p_asymptotic < 0.05).mean(), "share p < 0.05, permutation": (t.p < 0.05).mean()})
print("shuffled regions within countries, 50 runs (R1 table), mean:"); display(pd.DataFrame(null).mean().round(3).to_frame("mean"))'''),
    (M, "## R3 - regional hypotheses with region + year fixed effects, SEs clustered by country\nThe `RH` rows were specified in advance (regional analogues of the literature hypotheses); FDR is applied to them only. `S` rows were picked from the R1 survivors, so their p-values are not independent evidence."),
    (C, '''h = res["r3_region_panel"]
display(h[["id", "label", "expected_sign", "coef", "p", "q_value", "sign_agrees", "p_ex_irl", "n_obs", "n_regions"]].round(3))
ll = res["r3_lead_lag"]
display(ll.pivot_table(index=["id", "x", "y"], columns="lag", values="verdict", aggfunc="first"))'''),
    (M, "## R4 - is Italy's within-country slope different?\nSlope of y on x across regions after removing country means, Italy against all other countries, and the same difference for every other country as a placebo (p = share of countries with an equal or larger |difference|; the smallest possible p is 1 / number of countries). The last columns split Italy into North and South + Islands (exploratory, about 10 regions each)."),
    (C, '''s = res["r4_italy_slope"]
display(s[["id", "x", "y", "slope_italy", "slope_others", "slope_diff_italy", "p_diff", "n_countries", "slope_north", "slope_south"]].round(3))'''),
    (M, "## R5 - convergence of regional income\nAnnualised log growth of GDP per head (PPS) between 2005-07 and 2017-19 on the initial log level, country fixed effects, SEs clustered by country. beta < 0 means poorer regions catch up. Second block drops the richest region of each country."),
    (C, '''display(res["r5_convergence"].round(3))
lv0 = corr.cross_section(R[["gdp_pps_pc_eu27"]], "level", (2005, 2007), entity_level="geo")["gdp_pps_pc_eu27"]
lv1 = corr.cross_section(R[["gdp_pps_pc_eu27"]], "level", (2017, 2019), entity_level="geo")["gdp_pps_pc_eu27"]
g = pd.DataFrame({"ln0": np.log(lv0), "growth": (np.log(lv1) - np.log(lv0)) / 12 * 100}).dropna()
g["country"] = g.index.str[:2]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, cs, title in [(axes[0], ["IT"], "Italy"), (axes[1], ["DE", "ES", "FR", "PL"], "Germany, Spain, France, Poland")]:
    sub = g[g.country.isin(cs)]
    for c, grp in sub.groupby("country"):
        ax.scatter(grp.ln0, grp.growth, label=c, s=28)
        if len(grp) > 3: ax.plot(np.sort(grp.ln0), np.polyval(np.polyfit(grp.ln0, grp.growth, 1), np.sort(grp.ln0)), lw=1)
    ax.set_title(title); ax.set_xlabel("log GDP per head, 2005-07 (PPS)"); ax.legend(fontsize=8)
axes[0].set_ylabel("growth, % a year, 2005-07 to 2017-19")
plt.tight_layout(); plt.show()'''),
]

NOTEBOOKS["25_regional_correlation_discovery"].append((M, """## Takeaways (Eurostat NUTS2, 2015-19 unless stated; descriptive associations only)
- **The funnel (within countries, income removed):** 165 pairs survive false-discovery control in the 2015-19 levels (161 pass the leave-one-country-out and rank-vs-linear checks), 59 in the 2005-09 to 2015-19 changes, and 88 between countries (n about 27). Shuffling regions inside countries gives 0 discoveries in 3 runs. Most of the strongest pairs are labour-market pairs (long-term vs total unemployment, employment vs unemployment), so the cross-theme rows are more informative: AROPE with unemployment (rho 0.73, 137 regions), early school leaving with NEET (0.71), R&D personnel and physician density with tertiary attainment (about 0.6).
- **Permutation check (R2b):** retesting every pair with the within-country permutation p keeps 164 of the 165 levels survivors and 53 of the 59 changes survivors; the seven that drop out are weak (|rho| 0.14-0.21, permutation q 0.050-0.073). The check is needed: on shuffled regions the asymptotic p falls below 0.05 in 6.1% of pairs (0.12 discoveries per run, 50 runs), the permutation p in 5.0% (0 discoveries).
- **Pre-specified regional hypotheses (region + year fixed effects, country-clustered SEs):** 5 of 10 are significant at q < 0.05 and 4 go the expected way: early leaving and NEET, net migration and income per head, output and pay per hour (shares the hours denominator), unemployment and the investment rate. Broadband coverage and output per hour move in the *opposite* direction (coefficient -0.14), most likely because coverage saturates while productivity keeps changing. Graduate share and output per hour is borderline (q = 0.059); R&D intensity, graduates and female employment, and physicians and life expectancy show nothing.
- **Cross-sectional links do not survive region fixed effects.** All six AROPE pairs picked from the within-country screen are unrelated once region and year effects are added (p between 0.54 and 0.92; AROPE only starts in 2015), so differences between regions are persistent, not something that moves with time.
- **Lead-lag (1-3 years):** unemployment leads the investment rate (all lags); early leaving leads NEET at 2-3 years; old-age dependency leads old-age employment; output per hour leads the graduate share at 2-3 years (not the reverse); broadband-output, income-migration and output-pay hold in both directions, and R&D, graduates-female employment and physicians-life expectancy show no order.
- **Italy's slopes are the steepest of 14-18 countries in three relations** (graduate share and output per hour 1.18 vs 0.55; graduate share and female employment 3.15 vs 0.40; early leaving and NEET 1.62 vs 0.81), which is the smallest p the placebo ranking can give (1/17, 1/18, 1/14, so p = 0.06 to 0.07): suggestive, not significant, because Italy is one country. They are concentrated in the South and Islands (graduates and female employment: slope 3.03 there, -0.84 in the North; rho 0.79 vs -0.35), and about 10 regions each, so exploratory.
- **Convergence of regional income (2005-07 to 2017-19, country effects):** pooled beta -0.11 (p = 0.60, no convergence), Germany -1.13 (p < 0.001), Italy +0.47 (p = 0.21; +0.17 without the richest region), Spain +0.66, France +0.23, Poland +0.26 (none significant). Italy's South did not catch up with the North in this window.
- **Limits:** about 230 regions but only about 27 countries as clusters; NUTS codes mix vintages (coverage differs by indicator: 137 regions for AROPE, 235 for GDP-based ratios); LFS series carry many break flags (dropped for changes and panels, kept for single-window levels); output per hour uses current prices, not PPPs, so only within-country comparisons are reliable; no causal identification."""))


NOTEBOOKS["26_high_frequency_dynamics"] = [
    (M, """# High-frequency dynamics (monthly and quarterly Eurostat series, 2005-2026)
The annual panel cannot answer timing questions. Here: 12 monthly and 4 quarterly series (unemployment, HICP, industrial production, retail, 10-year yields, sentiment, GDP growth, employment, hours, labour cost), 31 countries and aggregates.
Safeguards: every series is **transformed to be approximately stationary** before any comparison (levels of yields, indices and prices trend and would correlate by accident); cross-correlations are **prewhitened**; panel tests control for the dependent series' own past with **country and month effects and country-clustered errors**; events are judged against a **placebo distribution from every other month of the same series** and corrected for multiple testing (BH-FDR).
**Predictive order and association only, never causal.** Code: `src/cifresignificative/analysis/hf.py`, `experiments.py`."""),
    (C, '''from cifresignificative.analysis import hf as H, experiments as ex
hf = H.load_highfreq()
res = ex.run_hf_all(hf)
display(hf.groupby("indicator").agg(countries=("iso3", "nunique"), observations=("value", "size"), first=("period", "min"), last=("period", "max")))
display(res["hf0_stationarity"].round(2))'''),
    (M, "## H1 - the Italy-Germany spread\nThe spread (10-year yield, Italy minus Germany) with Italian government changes and ECB decisions. Then the monthly change in the spread regressed on the previous month's (standardised) differentials with Germany: sentiment, change in unemployment, change in core inflation, industrial growth; Newey-West errors, by regime."),
    (C, '''sp = H.spread_series(hf)
ev = ex.italian_events()
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(sp.index, sp.values, color="C0", lw=1.6)
for d in ev[ev.type == "government"].date: ax.axvline(d, color="0.6", lw=0.7, ls=":")
for d, t in zip(ev[ev.type == "ecb"].date, ev[ev.type == "ecb"].event): ax.axvline(d, color="C1", lw=1, ls="--")
ax.set_ylabel("percentage points"); ax.set_title("Italy-Germany 10-year spread (dotted: new Italian governments; dashed orange: ECB decisions)")
plt.tight_layout(); plt.show()
print("peak:", sp.idxmax().date(), round(sp.max(), 2), "pp; latest:", sp.dropna().index[-1].date(), round(sp.dropna().iloc[-1], 2), "pp")
s = res["hf1_spread"]
display(s[s.country == "ITA"][["regime", "regressor", "coef_pp_per_sd", "se", "p", "q_value", "n", "r2"]].round(3))
print("Other countries, share of coefficients with q < 0.05 by country:"); display(s.groupby("country").apply(lambda g: (g.q_value < 0.05).mean()).round(2).to_frame("share").T)'''),
    (M, "## H2 - business-cycle lead-lag\nPanel Granger tests (country + month effects, own lags controlled, country-clustered errors), both directions, at 1, 3, 6 and 12 months (quarters for GDP). `x leads` / `y leads` = only that direction is significant at 5%."),
    (C, '''p = res["hf2_panel"]
display(p.pivot_table(index=["x", "y"], columns="lag", values="verdict", aggfunc="first"))
print("Italy against the other countries (prewhitened cross-correlation, lag 0-12 with the largest |r|; rank 1 = strongest):")
display(res["hf2_italy"].round(3))'''),
    (C, '''X, Y = H.transformed(hf, "m_esi"), H.transformed(hf, "m_industrial_production")
cc = {c: H.cross_corr(X[c], Y[c], 12) for c in ["ITA", "DEU", "FRA", "ESP"]}
fig, ax = plt.subplots(figsize=(8, 4))
for c, d in cc.items(): ax.plot(d.lag, d.r, marker="o", ms=3, label=c, lw=2.2 if c == "ITA" else 1)
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("lag in months (positive: sentiment leads production)")
ax.set_ylabel("prewhitened correlation"); ax.set_title("Economic sentiment vs industrial production growth"); ax.legend(); plt.tight_layout(); plt.show()'''),
    (M, "## H3 - from inflation to wages\nHourly labour cost growth on average inflation over the current and previous four quarters, controlling for unemployment; country and quarter effects. The coefficient is the cumulative pass-through (pp of wage growth per pp of inflation)."),
    (C, '''display(res["hf3_passthrough_panel"].round(3))
print("Italy against the other countries (country-by-country regression, 2005Q1-2021Q2):"); display(res["hf3_passthrough_italy"].round(3))
c = res["hf3_passthrough_countries"]; display(c.sort_values("pass_through", ascending=False).head(8).round(3))'''),
    (M, "## H4 - events\nChange in each series in the six months after a dated event minus the six months before, against a placebo distribution from all other months. 57 events (Italian governments, Italian policies, ECB decisions) x 4 series; q-values correct for the whole family. Events cluster at crisis turning points, and several policies share a month (identical rows), so read the table as a screen."),
    (C, '''e = res["hf4_events"]
print("tests:", len(e), " | p < 0.05:", int((e.p < 0.05).sum()), " | q < 0.10:", int((e.q_value < 0.10).sum()), " | q < 0.05:", int((e.q_value < 0.05).sum()))
display(e.head(12)[["label", "event", "type", "date", "change", "z", "p", "q_value"]].round(3))
print("share with p < 0.05 by event type:"); display(e.groupby("type").apply(lambda g: (g.p < 0.05).mean()).round(2).to_frame("share").T)'''),
]

NOTEBOOKS["26_high_frequency_dynamics"].append((M, """## Takeaways (Eurostat monthly and quarterly series, 2005-2026; predictive order and association only)
- **Data.** 16 series (12 monthly, 4 quarterly) for 23-31 countries and aggregates, 2005 to September 2026. HICP comes from the ECOICOP-2 dataset (`prc_hicp_minr`) because the old `prc_hicp_manr` stops in December 2025. Levels are mostly non-stationary (only 10% of countries pass ADF+KPSS for the inflation rate and 21% for labour-cost growth), while the transformed series pass in 68-100% of countries (sentiment levels are the weakest, 68-78%), so all tests use changes or growth rates.
- **The Italy-Germany spread** peaked at 5.19 pp in November 2011 (Monti's government took office that month) and stood at 0.89 pp in September 2026. The monthly change in the spread is hardly explained by the previous month's differentials with Germany (R2 1-16% by regime): nothing survives false-discovery control; the nominal best, industrial-growth gap (-0.025 pp per SD, q = 0.27 over the full sample and q = 0.16 since 2020), is small and does not hold across regimes. The same holds for Spain, France, Portugal and Greece (one coefficient in 80 has q < 0.05).
- **Business-cycle lead-lag (panel Granger tests, 27 countries).** Sentiment and industry confidence help predict industrial production growth one month ahead (and production predicts sentiment: both directions); at 3 months only industry confidence's *reverse* direction remains. Consumer confidence does not lead retail sales; retail sales lead consumer confidence at 3-6 months. Industrial production leads the change in unemployment only at 6 months, and sentiment and services confidence relate to unemployment in both directions at every lag. GDP growth leads employment and hours growth by 1-2 quarters (employment also predicts GDP at 2-4 quarters).
- **Italy's timing.** For Italy, the strongest prewhitened relation between industrial production growth and the change in unemployment is positive at lag 0 (r = 0.36, 2nd of 27 countries) because of 2020, when measured Italian unemployment fell in the lockdown months while output collapsed. Excluding March 2020 to June 2021 it becomes r = -0.07 at lag 0 and the strongest relation is a lead of production over unemployment of 6 months (r = -0.16), the same lead as the panel. Italy ranks 6th-13th of 26-27 on the sentiment and confidence relations, so nothing singles it out.
- **Wage pass-through.** Over 2005Q1-2021Q2 a one-point rise in average inflation (current and four previous quarters) goes with +0.44 pp of hourly labour-cost growth (p = 0.04, 28 countries); Italy's own estimate is 0.32, 19th of 27 (median 0.43). In 2021Q3-2023Q4, with quarter effects absorbing the common shock, the coefficient is 0.03 (p = 0.77): no detectable differential pass-through across countries in the energy-shock window.
- **Events (57 dated events x 4 series = 218 tests).** 16 have p < 0.05 (7%, close to the 5% expected by chance) and none has q < 0.10. The largest moves coincide with crisis turning points (Monti and the Salva Italia decree with unemployment +1.6 pp and the spread +1.3 pp over the following six months; the Draghi government and PNRR with sentiment +23 points, a post-pandemic rebound; the July 2022 ECB hike and TPI with the 10-year yield +1.8 pp), so the event dates cannot be separated from the cycle. No event type stands out (6-11% with p < 0.05 for policy, government and ECB events).
- **Limits:** associations and predictive order only; events are few, cluster in time and several share a month; HAC and country-clustered errors with 27 countries; seasonal adjustment by Eurostat; HICP from a reclassified dataset; GBR/USA/JPN are mostly absent from the monthly yield series."""))

NOTEBOOKS["27_daily_spreads_ratings"] = [
    (M, """# Daily Italy-Germany spread, rating changes and dated events (2005-2026)
Notebook 26 found no dated event that moves the *monthly* spread beyond chance; six-month windows are coarse. Here the same question at **daily** frequency: the Italy-Germany 10-year yield spread around Italian rating and outlook changes (S&P, Moody's, Fitch, DBRS), ECB decisions, Italian policies and government changes.
Data: daily closes from manual Investing.com exports (`data/external/source/daily_yields/`; Italy and Germany are the primary pair; Spain, France and Portugal give the peer spreads of section D5 and Greece the descriptive block D5b), cross-checked below against the monthly Eurostat yield. Rating changes are hand-collected (`data/external/rating_actions.csv`): every row is checked against the agency's own record of the action (`verified`; the saved agency pages and API output are in `data/external/source/rating_history/`; the `partly verified` and `unverified` labels, for dates backed only by aggregators or news, are no longer used).
Design: change in the spread over [t-1, t+h] trading days (h = 1, 5, 10), **divided by the recent volatility** of daily changes so the 2011-12 and 2020 events do not dominate; exact placebo p-value against all days whose window overlaps no event; one Benjamini-Hochberg family over every test; a direction-aligned **pooled test** for rating changes (downgrades should widen, upgrades narrow the spread); a **shuffled-date null** that must give about zero discoveries.
**Association only, never causal**: other news arrives on the same day, and agencies announce on pre-set review dates that markets may anticipate. Code: `src/cifresignificative/analysis/daily.py`, `experiments.py`."""),
    (C, '''from cifresignificative.analysis import daily as DY, hf as H, experiments as ex
dy = DY.load_daily()
res = ex.run_daily_all(dy)
x = DY.spread_daily(dy)
display(dy.groupby("iso3").agg(days=("value", "size"), first=("date", "min"), last=("date", "max"), min_yield=("value", "min"), max_yield=("value", "max")).round(2))
print("common days (spread):", len(x), x.index.min().date(), "to", x.index.max().date())
# anchor: monthly mean of the daily yield against the Eurostat monthly convergence yield
hf = H.load_highfreq()
for c in ["ITA", "DEU"]:
    m = H.wide_hf(hf, "m_long_term_yield")[c]
    d = dy[dy.iso3 == c].set_index("date").value.resample("ME").mean()
    j = pd.concat([m, d], axis=1, keys=["eurostat", "daily"]).dropna()
    print(c, "months:", len(j), "| correlation:", round(j.corr().iloc[0, 1], 4), "| median |difference| (pp):", round((j.eurostat - j.daily).abs().median(), 3))'''),
    (M, "## D0 - the daily spread and the dated events\nRating changes: red = downgrade or worse outlook, green = upgrade or better outlook (filled marker: two sources agree on the date; hollow: one source or conflicting dates). Dashed orange: ECB decisions."),
    (C, '''ra = ex.rating_actions()
ev = ex.italian_events()
fig, ax = plt.subplots(figsize=(11.5, 4.6))
ax.plot(x.index, x.values, color="C0", lw=0.9)
for d in ev[ev.type == "ecb"].date: ax.axvline(d, color="C1", lw=0.9, ls="--")
for _, r in ra.iterrows():
    if r.date in x.index:
        ax.scatter(r.date, x.loc[r.date], s=38, marker="v" if r.direction > 0 else "^", facecolors=("none" if r.evidence == "unverified" else ("#c0392b" if r.direction > 0 else "#2e8b57")),
                   edgecolors="#c0392b" if r.direction > 0 else "#2e8b57", zorder=3)
ax.set_ylabel("percentage points"); ax.set_title("Italy-Germany 10-year spread, daily (triangles: rating changes; dashed: ECB decisions)")
plt.tight_layout(); plt.show()
print("peak:", x.idxmax().date(), round(x.max(), 2), "pp; latest:", x.index[-1].date(), round(x.iloc[-1], 2), "pp; rating changes:", len(ra), "(", int((ra.direction > 0).sum()), "adverse,", int((ra.direction < 0).sum()), "favourable )")
print("rating changes by evidence:", ra.evidence.value_counts().to_dict(), "| dates outside the spread sample:", int((~ra.date.isin(x.index)).sum()), "(non-trading days snap to the next one)")'''),
    (M, "## D1 - every event, three horizons\nOne test per event date and horizon (events listed in two source files are merged). Raw change in pp and volatility-normalised z; q-values control the false-discovery rate over the whole family."),
    (C, '''e = res["d1_events"]
print("tests:", len(e), "| event dates:", e.date.nunique(), "| p < 0.05:", int((e.p < 0.05).sum()), "(about", round(0.05 * len(e), 1), "expected by chance) | q < 0.10:", int((e.q_value < 0.10).sum()), "| q < 0.05:", int((e.q_value < 0.05).sum()))
display(e.head(12)[["event", "type", "date", "h", "change_pp", "z", "p", "q_value"]].round(3))
print("share with p < 0.05 by event type:"); display(e.groupby("type").apply(lambda g: pd.Series({"tests": len(g), "share_p05": (g.p < 0.05).mean()})).round(2).T)
rt = e[e.type.str.contains("rating")]
print("rating-related tests:", len(rt), "| p < 0.05:", int((rt.p < 0.05).sum()), "| smallest p:", round(rt.p.min(), 3), "| smallest q:", round(rt.q_value.min(), 3))'''),
    (M, "## D2 - do rating changes move the spread in the expected direction?\nPooled over events, with the sign set by the direction of the change: mean of direction x z against the same statistic for random non-event days (one-sided p, H1: adverse changes widen and favourable changes narrow the spread). Robustness: volatility window 60 vs 20 days; all 40 changes (every date is `verified`, so there is a single evidence level; the code still compares levels if the file ever has several)."),
    (C, '''p = res["d2_ratings_pooled"]
display(p[["evidence", "vol_window", "h", "n_events", "mean_signed_z", "mean_signed_change_pp", "share_expected_sign", "p_one_sided", "q_value"]].round(3))
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for k, ((ev_, vw), g) in enumerate(p.groupby(["evidence", "vol_window"])):
    ax.errorbar(g.h + (k - (p.groupby(["evidence", "vol_window"]).ngroups - 1) / 2) * 0.35, g.mean_signed_z, yerr=1.96 * g.null_sd, fmt="o", capsize=3, label=f"{ev_}, {vw}-day vol")
ax.axhline(0, color="k", lw=0.6); ax.set_xticks([1, 5, 10]); ax.set_xlabel("horizon h (trading days after the event)")
ax.set_ylabel("mean direction x z"); ax.set_title("Pooled rating-change effect (bars: +-1.96 sd of the placebo)"); ax.legend(fontsize=7); plt.tight_layout(); plt.show()
print("smallest one-sided p:", p.p_one_sided.min().round(3), "| smallest q:", p.q_value.min().round(3), "| comparisons:", len(p))'''),
    (M, "## D2b - checks on the results above\nThe two largest per-event results and the pooled rating test without the crisis clusters."),
    (C, '''ecb = e[e.type.str.contains("ecb")]
print("ECB-related tests:", len(ecb), "on", ecb.date.nunique(), "dates | p < 0.05:", int((ecb.p < 0.05).sum()), "(all narrow the spread:", bool((ecb[ecb.p < 0.05].change_pp < 0).all()), "), q < 0.10:", int((ecb.q_value < 0.10).sum()))
display(ecb[ecb.p < 0.05][["event", "date", "h", "change_pp", "z", "p", "q_value"]].assign(event=lambda d: d.event.str[:48]).round(3))
w = dy.pivot(index="date", columns="iso3", values="value")
print("March 2020 lockdown window: yields (%) before and after the 9 March announcement:")
display(w.loc["2020-03-06":"2020-03-13"].assign(spread=lambda d: d.ITA - d.DEU).round(2))
allev = ex.daily_events()
rows = []
for name, sub in [("all", ra), ("without 2011-12", ra[ra.date >= "2013-01-01"]), ("without 2020-21", ra[(ra.date < "2020-01-01") | (ra.date >= "2022-01-01")])]:
    rows.append(DY.pooled_signed(x, sub, vol_window=60, exclude=allev.date).assign(subset=name, n_events=lambda d: d.n_events))
display(pd.concat(rows)[["subset", "h", "n_events", "mean_signed_z", "share_expected_sign", "p_one_sided"]].round(3))'''),
    (M, "## D3 - null check\n`n` random dates (as many as the dated events), the full event study with false-discovery control, repeated 200 times. The mean number of discoveries should be about zero."),
    (C, '''display(res["d3_placebo"].round(3))'''),
]

NOTEBOOKS["27_daily_spreads_ratings"] += [
    (M, """## D2c - does the pooled rating result depend on the evidence behind the date?
Descriptive sensitivity, not a new test family: the pooled test repeated for each distinct evidence level (`unverified` < `partly verified` < `verified`, where `verified` means the date is in the agency's own record). Earlier versions of this notebook found a one-day tilt only among dates backed by two aggregators; once the agencies' own pages replaced them, **every one of the 40 rows is `verified`**, so there is one level and nothing to compare. The table is kept so that the comparison reappears if rows with weaker evidence are added."""),
    (C, '''rs = ex.run_daily_rating_support(dy)
display(rs[["subset", "h", "n_events", "mean_signed_z", "share_expected_sign", "p_one_sided"]].round(3))
print("evidence levels:", rs.subset.unique().tolist(), "| comparisons:", len(rs), "| Bonferroni threshold for p < 0.05:", round(0.05 / len(rs), 4))'''),
]

NOTEBOOKS["27_daily_spreads_ratings"] += [
    (M, """## D5 - the same events on Italy's spread over Spain, France and Portugal
Does the Italy-Germany result hold when the reference is another large euro-area member? Same 93 dates, horizons, volatility normalisation and exact placebo p-values (`run_daily_peer_events`), with the Italy-**minus-peer** spread. Three families of 279 tests, one per peer (`q_peer`), and one of 837 over all peers (`q_value`); the Italy-Germany family of notebook section D1 is unchanged and stays the primary result. The direction-aligned pooled test for the 40 rating changes is repeated per peer (18 rows, `run_daily_peer_pooled`); Germany is shown beside it as a reference, not as part of the family. Each peer spread also gets its own shuffled-date null. No formal test of *differences* between peers and Germany is run: the table compares estimates and the placebo sd.
**Overlap:** the four spreads share Italy's yield and most events, so the families are not independent; the larger families only make the false-discovery control stricter."""),
    (C, '''pe, pp, ppl = res["d4_peer_events"], res["d5_peer_pooled"], res["d6_peer_placebo"]
d1 = res["d1_events"]
fam = pd.DataFrame({"ITA-DEU (primary)": {"tests": len(d1), "p < 0.05": int((d1.p < 0.05).sum()), "smallest q (own family)": d1.q_value.min()},
                    **{f"ITA-{k}": {"tests": len(g), "p < 0.05": int((g.p < 0.05).sum()), "smallest q (own family)": g.q_peer.min(), "smallest q (all peers)": g.q_value.min()} for k, g in pe.groupby("peer")}}).T
fam["expected by chance"] = (fam.tests * 0.05).round(1)
display(fam.round(3))
print("largest peer moves (p):"); display(pe.head(10)[["peer", "event", "h", "change_pp", "z", "p", "q_peer", "q_value"]].round(3))
print("pooled rating test, volatility window 60 (one-sided p; mde_z = smallest mean signed z reachable at p < 0.05, mde_z_family at Bonferroni over the 18 peer rows):")
display(pp[pp.vol_window == 60].set_index(["peer", "h"])[["n_events", "mean_signed_z", "mean_signed_change_pp", "share_expected_sign", "p_one_sided", "q_peer", "q_value", "null_sd", "mde_z", "mde_z_family"]].round(3))
print("shuffled-date null per peer:"); display(ppl.set_index("peer").round(3))
fig, ax = plt.subplots(figsize=(8, 3.6))
for k, peer in enumerate(["DEU", *ex.PEERS]):
    d = pp[(pp.peer == peer) & (pp.vol_window == 60)]
    ax.bar(np.arange(len(d)) + k * 0.2 - 0.3, d.mean_signed_z, width=0.18, label="ITA-" + peer + (" (reference)" if peer == "DEU" else ""))
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(["h = 1", "h = 5", "h = 10"]); ax.set_ylabel("mean signed z, 40 rating changes"); ax.legend(fontsize=8); ax.set_title("Pooled rating test by reference country (placebo sd about 0.16)")
plt.tight_layout(); plt.show()'''),
    (M, """## D5b - Greece (descriptive, no test family)
Greece-minus-Germany spread around the same Italian dates, as a contagion look, not part of any false-discovery family. Greek quotes need care (`data/external/source/daily_yields/README.md`): no data from late February to June 2005; in October 2011 to January 2012 the aggregator quotes another bond than the Eurostat convergence yield around the March 2012 restructuring (PSI); yields above 30% make the volatility scaling unstable. So the block is shown twice: `full`, and `after_psi` (events from April 2013 only, and no placebo day from March 2011 to March 2013). The Greek yield is **not** spliced or filled."""),
    (C, '''gr = res["d7_greece"]
ge = gr[gr.block == "events"]
tab = ge.groupby("variant").agg(tests=("p", "size"), p_below_005=("p", lambda v: int((v < 0.05).sum())), smallest_p=("p", "min"), placebo_days=("n_placebo", "median"))
tab["expected by chance"] = (tab.tests * 0.05).round(1)
display(tab.round(4))
print("largest Greek moves, after the PSI:"); display(ge[ge.variant == "after_psi"].sort_values("p").head(8)[["event", "h", "change_pp", "z", "p"]].round(3))
print("pooled rating test on the Greece-Germany spread (descriptive; one-sided p for the expected widening):")
display(gr[gr.block == "ratings_pooled"].set_index(["variant", "h"])[["n_events", "mean_signed_z", "share_expected_sign", "p_one_sided", "null_sd"]].round(3))'''),
]

NOTEBOOKS["27_daily_spreads_ratings"] += [
    (M, """## D4 - who holds Italian government debt? Bank holdings, two sources (descriptive only)
The spread question needs the holders of the debt. Two sources, neither like-for-like: **ECB SHSS** (`sources/ecb_shss.py`; quarterly, 2021-Q1 onwards; all Italian banks, debt securities, market value; banks against general government alone are *not* published, so the bank series is the residual `S1 - S12 - S1M` = government **plus non-financial corporations**, an upper bound) and the **EBA transparency exercise** (`sources/eba_sovereign.py`; semi-annual, 2019-12 onwards; 9-12 large banks; loans and securities; exposure to Italian central, regional and local government). With 9-11 common dates this is a description, not a test."""),
    (C, '''from cifresignificative.sources import ecb_shss, eba_sovereign
sh = ecb_shss.load().pivot(index="period", columns="series", values="value") / 1e3
eb = eba_sovereign.load().pivot(index="period", columns="measure", values="value")
nb = eba_sovereign.load().query("measure == 'net'").set_index("period").n_banks
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
a1.plot(sh.index, sh.it_fin_corps_gov, label="SHSS: all financial corporations, government debt", color="C0")
a1.plot(sh.index, sh.it_central_bank_gov, label="SHSS: central bank, government debt", color="C2")
a1.plot(sh.index, sh.it_banks_gov_nfc_residual, label="SHSS: banks, government + non-financial corporate debt (upper bound)", color="C1")
a1.plot(eb.index, eb.net, "o--", label="EBA: 9-12 large banks, net exposure to Italy (loans and securities)", color="C3")
a1.set_ylabel("EUR bn"); a1.legend(fontsize=8, loc="upper right"); a1.set_title("Holdings of Italian government debt by holder (EUR bn)")
q = x.resample("QE").last()
a2.plot(q.index, q.values, color="k"); a2.set_ylabel("spread, pp"); a2.set_title("Italy-Germany 10-year spread at quarter end")
plt.tight_layout(); plt.show()
j = pd.concat([eb.net.rename("eba_net"), nb.rename("n_banks"), sh.it_banks_gov_nfc_residual.rename("shss_residual")], axis=1).dropna()
j["ratio"] = j.eba_net / j.shss_residual
display(j.round(2))
d = j[["eba_net", "shss_residual"]].diff().dropna()
print("common dates:", len(j), "| ratio range:", round(j.ratio.min(), 2), "-", round(j.ratio.max(), 2), "| half-year changes with the same sign:", int((np.sign(d.eba_net) == np.sign(d.shss_residual)).sum()), "of", len(d))
print("SHSS 2021-Q1 to 2026-Q2 (bn):", {k: (round(sh[k].iloc[0]), round(sh[k].iloc[-1])) for k in ["it_fin_corps_gov", "it_central_bank_gov", "it_banks_gov_nfc_residual"]})
print("EBA net by date (bn):", eb.net.round(0).to_dict())'''),
]

NOTEBOOKS["27_daily_spreads_ratings"] += [
    (M, """## D4b - the Eurosystem's purchases of Italian government debt (PSPP and PEPP, monthly, descriptive only)
`sources/eurosystem_holdings.py` reads the two ECB country files (PSPP 2015-03 on, PEPP 2020-03 on; manual download, `data/external/source/eurosystem_holdings/`). They give **monthly net purchases** (net of redemptions, EUR million, amortised cost) and the average maturity of the holdings. The **stock** below is the cumulative sum of net purchases: it adds up to the ECB's own cumulative column within rounding, and at end-2025 it equals the figures in the Treasury's 2025 debt report (PEPP about EUR 247bn, PSPP 312.2bn), so the two are the same basis. It is an amortised-cost stock, not a market value, so it is compared with the ECB SHSS central-bank holdings (market value, all domestic government debt held by the central bank, so it also holds anything outside the two programmes) only as a ratio. No test."""),
    (C, '''from cifresignificative.sources import eurosystem_holdings as EH
eh = EH.load()
itl = eh[eh.country == "Italy"]
stock = itl[itl.measure == "stock"].pivot(index="period", columns="programme", values="value") / 1e3
stock["APP"] = stock.PSPP + stock.PEPP
wam = itl[itl.measure.str.startswith("wam")].pivot_table(index="period", columns=["programme", "measure"], values="value")
chk = EH.cumulative_check().query("country == 'Italy'").set_index("programme")[["sum_monthly", "ecb_cumulative", "diff"]]
print("monthly net purchases summed vs the ECB cumulative column, Italy (EUR million):"); display(chk)
qe = stock.resample("QE").last().join(sh.it_central_bank_gov.rename("shss_central_bank"), how="inner")
qe["shss / APP stock"] = qe.shss_central_bank / qe.APP
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
a1.stackplot(stock.index, stock.PSPP, stock.PEPP, labels=["PSPP stock (cumulative net purchases)", "PEPP stock"], colors=["C0", "C2"], alpha=0.8)
a1.plot(sh.index, sh.it_central_bank_gov, "o--", color="C3", label="ECB SHSS: central bank, government debt (market value)")
a1.set_ylabel("EUR bn"); a1.legend(fontsize=8, loc="upper left"); a1.set_title("Eurosystem purchases of Italian government debt (EUR bn, amortised cost)")
for (prog, m), v in wam.items():
    if m == "wam_portfolio": a2.plot(v.index, v.values, label=f"{prog} holdings")
a2.plot(wam[("PSPP", "wam_universe")].dropna().index, wam[("PSPP", "wam_universe")].dropna().values, "k:", label="PSPP eligible universe")
a2.set_ylabel("years"); a2.legend(fontsize=8, ncol=3); a2.set_title("Remaining weighted average maturity")
plt.tight_layout(); plt.show()
display(stock.loc[["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31", "2026-09-30"]].round(1))
print("peak: PSPP", stock.PSPP.idxmax().date(), round(stock.PSPP.max(), 1), "| PEPP", stock.PEPP.idxmax().date(), round(stock.PEPP.max(), 1), "| both", stock.APP.idxmax().date(), round(stock.APP.max(), 1))
print("ratio SHSS central bank / APP stock, quarter ends:"); display(qe[["APP", "shss_central_bank", "shss / APP stock"]].round(3).T)'''),
]

NOTEBOOKS["27_daily_spreads_ratings"].append((M, """## Takeaways (daily 10-year yields, 2005-2026; association only, never causal)
- **Data.** Daily closes for Italy (5,613 trading days) and Germany (5,609), 3 January 2005 to 14 October 2026; 5,597 common days (Spain, France, Portugal and Greece are loaded too, 5,508-5,634 days each, used in sections D5 and D5b). The exports carry indicative weekend quotes (370 Italian and 182 German rows), dropped. Monthly means of the daily series track the Eurostat monthly convergence yield (correlation 0.9995 for Italy and 0.9999 for Germany, median gap 0.013 and 0.036 pp). The spread peaked at 5.56 pp on 9 November 2011 and stood at 1.08 pp on 14 October 2026.
- **Rating changes.** 40 changes by S&P, Moody's, Fitch and DBRS (23 adverse, 17 favourable), September 2011 to January 2026, **all `verified`**: each date is in the agency's own record of the action (Moody's and Fitch rating-history pages, DBRS and S&P document lists). That check corrected one date (Moody's outlook change: 5 August 2022, not 8 August) and added twelve rows (Moody's 13 July 2012, 14 February 2014, 7 December 2016; Fitch 8 March 2013, 25 April 2014, 21 October 2016; DBRS 9 November 2011, 8 August 2012, 7 March 2013, 27 March 2015; S&P 10 July 2013 and 5 December 2014). The S&P 2013 downgrade is dated 10 July (the publication of its correction, 13:41 EDT). Scope Ratings is not covered. Watches and reviews, confirmations and everything before 19 September 2011 are not rating changes in this file.
- **Events (279 tests: 93 event dates x 3 horizons).** 17 have p < 0.05, against about 14 expected by chance, and none has q < 0.10 (smallest q 0.163). The largest move is the national COVID-19 lockdown on 9 March 2020: the spread rose 0.83 pp over the next five trading days (z = 6.5), as the Italian yield went from 1.08% to 1.41% in one trading day while the Bund yield also fell; its q-value over the 279 tests is 0.163. Four of 15 ECB-related tests have p < 0.05, all with a narrowing spread (PEPP on 18 March 2020: -0.81 pp at one day and -0.98 pp at five days; "whatever it takes" on 26 July 2012: -0.65 pp; OMT on 6 September 2012: -0.57 pp), but their smallest q is 0.430 and they sit on only three dates.
- **Rating changes show no consistent daily move.** Of 120 rating-related tests, 3 have p < 0.05 (smallest p 0.002, q 0.163): two favourable outlook changes in October 2021 after which the spread *widened* (DBRS on 29 October, +0.17 pp at one day; S&P on 22 October, +0.18 pp at five days) and the Fitch downgrade of 21 April 2017 after which the spread *narrowed* by 0.18 pp at one day (z = -2.4, p 0.033). Pooled with the sign of each change, all 40 events give a mean signed z of 0.09 at one day (one-sided p 0.29; 25 of 40 in the expected direction), -0.04 at five days and 0.02 at ten days (p 0.62 and 0.45; q = 0.63 over the 6 pooled tests, 0.07 and p 0.35 at one day with a 20-day volatility window), and it stays non-significant without the 2011-12 events (one-day p 0.46) or without the 2020-21 events (p 0.18). **The earlier one-day hint (14 `partly verified` dates: z 0.57, p 0.021) did not survive added dates** and has not come back with every date now verified: it was a small-sample result and should not be quoted.
- **Evidence levels (D2c).** An earlier version found a one-day tilt only among the 3 S&P dates backed by two aggregators (z 1.32, p 0.015 among 12 comparisons, not significant after correction). Once the agencies' own records replaced the aggregators, all 40 rows are verified, there is one evidence level and the tilt cannot be tested; it should not be quoted.
- **Calibration.** With 93 random dates (as many as the real event dates) and the same false-discovery control, 200 repetitions give 0.02 discoveries per repetition on average (1% of repetitions find one), so the procedure does not manufacture discoveries. Power is limited: with 40 rating changes the placebo sd of the pooled one-day z is 0.16, so an average signed z below about 0.27 (1.645 sd, one-sided 5%) cannot be told from noise.
- **Who holds the debt (D4, descriptive, no test: 9 common dates).** ECB SHSS (2021-Q1 to 2026-Q2): financial corporations' holdings of Italian government debt fell from EUR 1,535bn to 1,242bn and the central bank's from 671bn to 512bn; banks' holdings (government plus non-financial corporate debt, an upper bound because banks against government alone are not published) were 438bn at the start and 424bn at the end, with a low of 361bn in 2024-Q2. EBA (9-12 large banks, loans and securities): net exposure to Italy went from 291bn (2021-06) to 243bn (2024-12) and 262bn (2025-06). The two track each other: EBA is 0.66-0.70 of the SHSS residual on all 9 common dates and 6 of 8 half-year changes have the same sign. The 2019-12 (180bn, 11 banks) and 2020-12 (255bn, 9 banks) EBA rounds have a different bank sample, so their level is not a change. The sources differ in scope, so the ratio is consistent with EBA covering about two thirds of the SHSS aggregate but does not prove it.
- **Eurosystem purchases (D4b, descriptive, no test).** The ECB country files give Italy's PSPP and PEPP net purchases monthly. Their cumulative sum equals the ECB's own cumulative column within 1 million, and at end-2025 it is EUR 312.2bn for PSPP and 246.8bn for PEPP, the same figures as the Treasury's 2025 debt report (so the two sources are on one basis: amortised cost, net of redemptions). The combined stock peaked at EUR 739.3bn in July 2022 (PSPP 450.2bn; PEPP peaked later, 293.8bn in July 2023) and was 497.7bn in September 2026 (PSPP 278.0bn, PEPP 219.8bn), 241.6bn lower. Against the ECB SHSS central-bank holdings (market value, all domestic government debt) the ratio runs from 1.16 in 2021-Q1 down to 0.89 in late 2022 and up to 0.99 in 2026-Q2: the two series move together but are not the same quantity. The average remaining maturity of the holdings is 7.4 years for both programmes in September 2026 (eligible universe: 7.8 PSPP, 7.0 PEPP). The stock is not linked to the spread in any test.
- **Peer spreads (D5) and Greece (D5b).** The result does not depend on the reference country. Over Spain, France and Portugal, 16, 16 and 18 of 279 tests have p < 0.05 (about 14 expected; Germany 17); the smallest q is 0.631, 0.217 and 0.564 within each peer's family and 0.652 to 0.695 over all 837 tests. The largest moves are again the lockdown and the PEPP announcement, plus the DBRS favourable outlook change of 29 October 2021, after which the spread over France and Spain *widened* at one day (z 5.9 and 4.3; the same date is among the three p < 0.05 rating tests over Germany). Pooled over the 40 rating changes, the one-day mean signed z is 0.33 (Spain), 0.09 (France) and -0.01 (Portugal), against 0.09 for Germany; only Spain's has a raw one-sided p below 0.05 (0.028) and it has q 0.085 within Spain's six rows and 0.254 over the 18 peer rows. The smallest mean signed z that can reach p < 0.05 is 0.25 to 0.29 (0.43 to 0.49 with a Bonferroni correction over the 18 rows), so only a large aligned effect would show. Shuffled dates give 0.01 to 0.025 discoveries per repetition for each peer. Greece (descriptive, outside every family): 25 of 276 tests have p < 0.05 with all dates (about 14 expected) and 23 of 207 for events after March 2013 with no placebo day in 2011-13 (about 10); the pooled rating test points the opposite way from widening (mean signed z -0.26 at one day over 40 changes, -0.42 over 30 after the restructuring: the Greek spread tends to narrow around adverse Italian rating news, one-sided test, no two-sided test run). Not evidence of contagion or its absence.
- **Limits.** Daily yields exist for six countries but nothing separates Italian from euro-area news, and the peer families (D5) share Italy's yield and the events with the primary one, so they are not independent evidence; the holder data are short (SHSS 22 quarters, EBA 9-11 rounds) and are not linked to the spread statistically; yields come from a commercial aggregator checked against Eurostat only at monthly level; agencies publish on calendar dates known in advance and markets may price the outcome beforehand, which biases daily windows toward zero; other news arrives on the same day (the March 2020 events are within nine days of each other); the March 2020 event windows overlap one another, and all event windows are excluded from the placebo days; rating dates are partly verified at best. Results are descriptive associations, never causal effects.
"""))

NOTEBOOKS["28_industry_panel"] = [
    (M, """# Industry panel: does growth load on ICT- or intangible-intensive industries, and where is Italy?
Country-level screens rest on about 30 countries. Here the unit is **country x industry** (about 57 market industries at 2-digit level per country, `sectors64`), which gives thousands of observations, while the *country* remains the unit of judgement.
Design (Rajan-Zingales style): outcome = log growth per year of real value added per hour (mean 2005-07 to mean 2017-19; early and late halves as split-sample checks), winsorised at 1% / 99%. Industry intensity = contribution of ICT (or intangible) capital to labour productivity growth in the industry's EU KLEMS section, **median over the other countries, 1996-2004** (Italy left out, and before the outcome window). For each country: slope of growth on intensity.
Inference: the intensity varies only between ~14 sections, so a clustered t-test over-rejects (a shuffled-intensity check showed 15-20% false positives); p-values come from **permuting the intensity across sections**, and Italy is judged by its **rank among countries** (placebo ranking), never by a clustered test. Coverage: hours at 2-digit level are missing for several countries (Germany, France, Belgium, Portugal and others have 15 industries or fewer), so about 19 countries enter.
**Association only, never causal.** Code: `src/cifresignificative/analysis/industry_panel.py`, `experiments.run_industry_all`."""),
    (C, '''from cifresignificative.analysis import industry_panel as IP, experiments as ex
ek = pd.read_parquet(PROCESSED / "euklems_growth.parquet")
res = ex.run_industry_all()
S, rank, loo = res["slopes"], res["italy_rank"], res["italy_loo_section"]
for k in IP.INTENSITIES: print(k, "intensity by KLEMS section (median of other countries, 1996-2004, pp):", IP.intensity(ek, IP.INTENSITIES[k]).round(2).to_dict())
f = S[S.window == "full"]
print("countries:", f.groupby("intensity").size().to_dict(), "| industries per country (median):", int(f.n_industries.median()))
print("countries with p_perm < 0.05, by window and intensity:"); display(S.groupby(["window", "intensity"]).p_perm.agg(lambda p: int((p < 0.05).sum())).unstack())
print("Italy's placebo rank:"); display(rank.round(3))
print("Italy leave-one-section-out (ICT): slope range", loo[loo.intensity == "ict"].italy_slope.min().round(3), "to", loo[loo.intensity == "ict"].italy_slope.max().round(3), "| rank range", int(loo[loo.intensity == "ict"].rank_from_top.min()), "to", int(loo[loo.intensity == "ict"].rank_from_top.max()))
print("Italy leave-one-section-out (intangible): slope range", loo[loo.intensity == "intangible"].italy_slope.min().round(3), "to", loo[loo.intensity == "intangible"].italy_slope.max().round(3), "| rank range", int(loo[loo.intensity == "intangible"].rank_from_top.min()), "to", int(loo[loo.intensity == "intangible"].rank_from_top.max()))
from cifresignificative.analysis import corr
q = corr.fdr(S.rename(columns={"p_perm": "p"}))
print("tests:", len(q), "| BH over all of them: smallest q", round(q.q_value.min(), 3), "| survivors:", int(q.significant.sum()), "| p < 0.05:", int((q.p < 0.05).sum()), "(about", round(0.05 * len(q), 1), "expected by chance)")
print("calibration (shuffled intensity, share of countries with p_perm < 0.05, expect ~0.05):", IP.placebo_null(ek, n_rep=40)["mean_share_p_below_005"].__round__(3), IP.placebo_null(ek, n_rep=40, intensity_name="intangible")["mean_share_p_below_005"].__round__(3))'''),
    (C, '''fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=False)
for ax, k in zip(axes, IP.INTENSITIES, strict=True):
    d = f[f.intensity == k].sort_values("slope")
    cols = [ITALY_C if c == ITALY else "#8a8a85" for c in d.iso3]
    ax.barh(d.iso3, d.slope, color=cols, xerr=1.96 * d.se, error_kw={"lw": 0.8, "ecolor": "#bbb"}); ax.axvline(0, color=INK2, lw=0.8)
    ax.axvline(d.slope[d.iso3 != ITALY].median(), color="#eda100", ls="--", lw=1, label="median of the others")
    ax.set_title(f"Growth on {k} intensity"); ax.set_xlabel("slope (log growth per year per pp of intensity)"); ax.legend(fontsize=7)
fig.suptitle("Country slopes, industries 2005-07 to 2017-19 (bars: clustered 95% band, indicative only)", x=0.01, ha="left", fontsize=11); fig.tight_layout(); save(fig, "28_industry_slopes")'''),
]
NOTEBOOKS["28_industry_panel"].append((M, """## Takeaways (EU KLEMS intensities, Eurostat 2-digit industries, 2005-2019; descriptive associations only)
- **Coverage.** 19 countries in the full window (20 in the late half), 57 industries each for most (Luxembourg 26, Malta 25, Ireland 45, Sweden 49); Germany, France, Belgium, Portugal and others are out because hours worked are missing at 2-digit level. The industry panel has thousands of rows, but the intensity varies across only ~14 sections, so the effective sample for each slope is small.
- **No country-level pattern survives.** 116 country-window-intensity tests give 13 with permutation p < 0.05 (about 5.8 expected by chance); with false-discovery control over all of them the smallest q is 0.232 and none survives. Spain has the clearest negative ICT slope in the full window (p 0.008; also negative in the early half, but not in the late one) and Sweden and the Czech Republic positive intangible slopes (p 0.010 and 0.034; Czech and Swedish slopes recur in the late half), all below the discovery threshold.
- **Italy sits in the middle.** ICT slope -0.156 against a median of -0.097 for the others (rank 13 of 19 from the top, one-sided p that Italy is this low 0.368); intangible slope 0.019 against 0.023 (rank 12 of 19, p 0.421). Early and late halves give ranks 8 and 12 (ICT) and 7 and 16 (intangible): no stable position. Dropping any one section leaves Italy's rank between 9 and 14 (ICT) and 9 and 13 (intangible).
- **Calibration.** With the intensity shuffled across sections, 3.2% (ICT) and 4.1% (intangible) of country slopes have p < 0.05, so the permutation test is not inflated; the clustered t-test was (15-20%), which is why it is shown only as an indicative band.
- **Reading.** Italy's weak productivity growth is not concentrated in industries that rely more on ICT or intangible capital in other countries: the pattern is not distinguishable from the other countries' slopes. This does not rule out a firm-level or within-section story (notebooks 13 and 23 show where, not why). The benchmark intensity comes from 14 coarse sections, EU KLEMS capital contributions are an imperfect proxy for technology dependence, and R&D or regulation-based intensities were not available.
"""))

NOTEBOOKS["29_latent_structure"] = [
    (M, """# Latent structure of the national indicators (PCA)
Notebook 24 ranks indicator pairs; with ~85 indicators and 27 countries a few dimensions summarise most of the co-variation. Here: principal components of the **2015-19 country x indicator table** (EU27, size-scaled series and near-duplicates dropped, indicators with at least 80% country coverage, z-scores winsorised at 3 SD, missing = mean).
Checks: **parallel analysis** (how many components beat a shuffled-column null), **bootstrap over countries** and **split-half (2005-14 vs 2015-24) congruence** of the loadings, and **Italy's position over time**: later and earlier windows are projected onto the same axes, and Italy is judged by its rank among countries (placebo ranking), never as a cluster.
**Descriptive only**: components summarise co-variation; they are not causes, and the axis names below are my reading of the top loadings. Code: `src/cifresignificative/analysis/latent.py`, `experiments.run_latent_all`."""),
    (C, '''from cifresignificative.analysis import latent as LT, corr, experiments as ex
res = ex.run_latent_all(panel)
sm = res["summary"]
print("countries:", int(sm.n_countries[0]), "| indicators:", int(sm.n_indicators[0]), "| components retained by parallel analysis:", int(sm.n_retain[0]))
display(sm[["component", "eigenvalue", "null_p95", "explained"]].round(3).head(6))
print("cumulative explained, retained components:", round(sm.explained.head(int(sm.n_retain[0])).sum(), 3))
L_ = res["loadings"].set_index("indicator")
for c in L_.columns:
    print(c, "top loadings:", ", ".join(f"{i} ({v:+.2f})" for i, v in L_[c].reindex(L_[c].abs().sort_values(ascending=False).index).head(8).items()))
print("bootstrap stability (cosine with the full-sample loadings):"); display(res["bootstrap"].round(2))
print("split-half congruence (2005-14 vs 2015-24):"); display(res["split_half"].round(2))'''),
    (C, '''it = res["italy"]
print("Italy's score, median of the other countries and rank (1 = highest), by window:"); display(it.round(2))
sc = res["scores"].set_index("iso3")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
for c in sc.index: ax.scatter(sc.loc[c, "PC1"], sc.loc[c, "PC2"], color=ITALY_C if c == ITALY else "#8a8a85", s=40 if c == ITALY else 18, zorder=3 if c == ITALY else 2); ax.annotate(c, (sc.loc[c, "PC1"], sc.loc[c, "PC2"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Countries on the first two components, 2015-19")
ax = axes[1]
for c, col in zip(["PC1", "PC2", "PC3"], ["#2a78d6", "#eda100", "#1baf7a"], strict=False):
    d = it[it.component == c]
    if len(d): ax.plot(d.window, d.italy - d.median_others, marker="o", color=col, label=f"{c}: Italy minus median of the others")
ax.axhline(0, color=INK2, lw=0.8); ax.legend(fontsize=7); ax.set_title("Italy's distance from the others on fixed axes")
fig.tight_layout(); save(fig, "29_latent_structure")'''),
]
NOTEBOOKS["29_latent_structure"].append((M, """## Takeaways (27 EU members, 85 indicators, 2015-19 axes; descriptive only)
- **Three dimensions, half the variance.** Parallel analysis retains 3 components (eigenvalues 20.2, 12.8, 8.0 against shuffled-null 95th percentiles of 7.3, 6.5, 6.0; the fourth, 5.4, is below its null 5.5); together they explain 50.8% of the variance (25.0%, 15.9%, 9.9%).
- **What they look like (my reading of the top loadings).** PC1: governance, wages, R&D and education spending (control of corruption, hourly labour cost, gross earnings, regulatory quality, R&D, spending per student), a prosperity-and-institutions axis (recorded theft loads on it too, which probably reflects reporting rather than more crime). PC2: debt, interest, youth and total unemployment against employment and graduate employment, a debt-and-labour-market weakness axis. PC3: emigration, immigration and permits against tax wedge and old-age dependency (mobility versus ageing); it is the weakest and least stable.
- **Stability.** Bootstrapping countries, the median best cosine with the full-sample loadings is 0.91 (PC1), 0.82 (PC2) and 0.72 (PC3), with 5th percentiles 0.71, 0.58 and 0.17: PC1 is solid, PC2 acceptable, PC3 can be replaced by a different mix in some resamples. Fitting 2005-14 and 2015-24 separately (81 shared indicators), the best cosines are 0.95, 0.81 and 0.89, in the same order.
- **Italy.** Measured on fixed 2015-19 axes, Italy minus the median of the other countries on PC2 is +5.4 (2005-09), +7.3 (2010-14), +8.2 (2015-19) and +6.2 (2020-24): Italy ranks 1st, 2nd, 2nd and 2nd of 27 on the debt-and-labour-market-weakness axis in every window, the gap widening after the 2008-12 crises and narrowing after 2020. On PC1 Italy is below the median in each window (gap -1.1, -1.2, -2.0, -1.6; rank 17-18 of 27), i.e. mid-table to lower, not an outlier; on PC3 the gap is -1.7, -1.2, -1.0, -1.5 (rank 22-26 of 27 in all windows).
- **Limits.** 27 countries against 85 indicators: the axes are estimated with a few degrees of freedom, and the bootstrap shows it. Scores of earlier or later windows are projected on 2015-19 axes, so common drift (median scores move across windows) is removed only by taking the difference to the other countries; the 2005-09 window covers 74% of the indicators for Italy and missing values are set to the mean, which pulls its scores towards zero. Indicator selection (coverage, duplicates) shapes the axes. Axis names are interpretation, not results.
"""))

NOTEBOOKS["30_income_distribution"] = [
    (M, """# Income distribution: how does Italy's spread of income compare with the EU27?
Until now inequality was one Gini and a few poverty rates. Here the **shape**: shares of equivalised disposable income held by each quintile (and the bottom and top decile), the S80/S20 ratio, and the Ginis **before and after** social transfers (EU-SILC through Eurostat: `ilc_di01`, `ilc_di11`, `ilc_di12`, `ilc_di12b`, `ilc_di12c`; table `data/processed/income_dist.parquet`).
Method: country means over three windows (2005-09, 2015-19, 2021-25; breaks `b` and provisional `p` values dropped), Italy ranked among the 27 members, and judged by a **placebo ranking** (each country's gap to the median of the others, ranked against Italy's gap; `p_abs` = share of countries with an absolute gap at least as large), not a test with Italy as a cluster. Years are Eurostat survey years; incomes refer mostly to the previous calendar year. The disposable-income Gini starts in 2014, so redistribution can be measured only from 2015.
**Descriptive only**: this shows where Italy sits, not why. Code: `src/cifresignificative/analysis/income_dist.py`."""),
    (C, '''from cifresignificative.analysis import income_dist as ID
inc = ID.load_income()
W = ID.wide(inc)
print("rows:", len(inc), "| countries:", inc[inc.iso3.isin(EU27_ISO3)].iso3.nunique(), "| years:", inc.year.min(), "-", inc.year.max(), "| flags dropped:", sorted(ID.DROP_FLAGS))
tab = ID.italy_table(W)
show = tab.pivot(index="measure", columns="window", values=["value", "median_others", "rank_from_top", "p_abs"]).round(2)
display(show)
print("Italy's quintile shares minus the median of the others (pp), by window:")
display(pd.DataFrame({w: ID.shape_gap(W, win).gap for w, win in ID.WINDOWS.items()}).round(2))
print("Change in the window mean 2005-09 -> 2021-25 (Italy vs placebo ranking, 1 = largest rise):"); display(ID.change_table(W, (2005, 2009), (2021, 2025)).dropna().round(2))
print("calibration (shuffled countries, share with p_abs < 0.05):", ID.shuffled_null(W, (2015, 2019)))'''),
    (C, '''fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axes[0]; w = 0.27
for k, (name, win) in enumerate(ID.WINDOWS.items()):
    g = ID.shape_gap(W, win).gap; ax.bar(np.arange(5) + (k - 1) * w, g.values, w, label=name, color=["#9ec5f0", "#2a78d6", "#0b3d7a"][k])
ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(range(5), ["Q1\\n(poorest)", "Q2", "Q3", "Q4", "Q5\\n(richest)"]); ax.set_ylabel("pp of income, Italy minus median of the others"); ax.set_title("Where Italy's shares differ"); ax.legend(fontsize=7)
ax = axes[1]; s = W["s80_s20"].unstack("iso3")
for c in s: ax.plot(s.index, s[c], color="#d0d0cc", lw=0.7)
oth = s.drop(columns=ITALY).median(axis=1); ax.plot(s.index, oth, color="#eda100", ls="--", lw=1.6, label="median of the others"); ax.plot(s.index, s[ITALY], color=ITALY_C, lw=2, label="Italy"); ax.set_title("S80/S20"); ax.legend(fontsize=7)
ax = axes[2]; T = ID.window_table(W, ID.WINDOWS["2015-19"]).dropna(subset=["gini_pre_pens_excl", "gini_disp"])
for c in T.index: ax.scatter(T.loc[c, "gini_pre_pens_excl"], T.loc[c, "gini_disp"], color=ITALY_C if c == ITALY else "#8a8a85", s=44 if c == ITALY else 16, zorder=3 if c == ITALY else 2); ax.annotate(c, (T.loc[c, "gini_pre_pens_excl"], T.loc[c, "gini_disp"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
lim = [T[["gini_pre_pens_excl", "gini_disp"]].min().min() - 1, T[["gini_pre_pens_excl", "gini_disp"]].max().max() + 1]; ax.plot(lim, lim, color=INK2, lw=0.7, ls=":")
ax.set_xlabel("Gini before social transfers (pensions kept as income), 2015-19"); ax.set_ylabel("Gini of disposable income"); ax.set_title("How much transfers other than pensions reduce inequality")
fig.tight_layout(); save(fig, "30_income_distribution")'''),
]
NOTEBOOKS["30_income_distribution"].append((M, """## Takeaways (EU-SILC via Eurostat, survey years 2005-2025; descriptive, cross-country)
- **The shape: thin bottom, heavy top, middle on par.** Italy's bottom quintile holds 1.7 pp less of income than the median of the other members in 2005-09 (7.3% vs 9.0%), 1.9 pp less in 2015-19 (6.6% vs 8.5%) and 1.2 pp less in 2021-25 (7.2% vs 8.4%); the top quintile holds 2.0, 1.7 and 2.2 pp more (39.3-39.5% vs 37.2-37.8%); the middle quintile is within 0.2 pp. S80/S20 is 5.4, 6.0 and 5.5 against a median of 4.1, 4.5 and 4.5 (Italy ranks 9th of 26, then 6th of 27 twice, 1 = most unequal); the Gini of disposable income (available from 2014) is 32.9 vs 29.4 in 2015-19 and 32.1 vs 29.3 in 2021-25 (5th highest).
- **But no placebo ranking singles Italy out.** Italy's gap to the median is ranked against every country's own gap: the share of countries with a gap at least as large (`p_abs`) is 0.22-0.37 for S80/S20 and 0.48 for the Gini; the smallest value in the table is 0.07 for the bottom decile in 2015-19 (Italy 26th of 27, 2.0% of income vs 3.2%), which does not stand out given the number of cells read. Italy is moderately more unequal than the typical member, with several others further out (on S80/S20 in 2015-19: Bulgaria, Romania, Lithuania, Latvia and Spain; on the disposable Gini in 2021-25: Bulgaria, Lithuania, Latvia and Portugal).
- **Little change in shape since 2005-09.** Italy's rise in S80/S20 (+0.09 between the window means) and in each quintile share is ordinary against the other countries (all `p_abs` >= 0.35; 26 countries with both windows); the early-2010s deterioration and the 2022-25 improvement roughly offset in the means (S80/S20 6.3 in 2016, 5.1 in 2025).
- **Redistribution is where Italy is weakest, depending on the definition.** Counting pensions as income (so only other transfers redistribute), the Gini falls 6.7% from before transfers to disposable income in 2015-19, the lowest of the 27 (median of the others 13.3%, range 6.9-28.1%; `p_abs` 0.22); in 2021-25 it is 11.6% (16th, median 12.4%), after the pandemic-era transfers. Counting pensions as transfers, the fall is 32% against 42% in 2015-19 (22nd of 27) and 34% against 39% in 2021-25 (19th). The pre-transfer Gini (pensions as income) rose 2.5 points from 2005-09 to 2021-25 against a median change of -0.6 (4th largest rise of 26, `p_abs` 0.35).
- **Limits.** Survey years, not income years; the disposable-income Gini starts in 2014, so redistribution is only measured from 2015; pre-transfer measures are conventions (pensions in or out) rather than a counterfactual without the welfare state, since behaviour would differ; EU-SILC under-covers the very rich and the very poor (top shares from tax data would be higher); Italy's 2022 pre-transfer values carry a break flag and are dropped; about a hundred cells were read, so isolated p_abs values near 0.05 are expected by chance. Descriptive: it says where Italy sits, not why.
"""))

NOTEBOOKS["31_self_employment"] = [
    (M, """# Self-employment: how large is it in Italy, where, and how do the self-employed fare?
Notebook 30 described the income distribution; the next question is **who the self-employed are**. Eurostat's labour force survey (`lfsa_esgan`, `lfsa_esgan2`, `lfsa_egan2`) gives the share of employment that is self-employed (employers and own-account workers without employees), by sector and age; EU-SILC (`ilc_li04`) gives at-risk-of-poverty rates by activity status (`NSAL` = employed persons except employees: self-employed plus contributing family workers). Table `data/processed/selfemp.parquet`, code `src/cifresignificative/analysis/selfemp.py`.
Method: window means (breaks `b` and provisional `p` values dropped), Italy ranked among the 27 members by a **placebo ranking** (Italy's gap to the median of the others against every country's own gap; `p_abs` = share of countries with a gap at least as large; with 27 countries the smallest possible value is 1/27 = 0.04, so 0.07 means the second largest gap), and an exact **shift-share decomposition** of the sector mix against the pooled other countries.
**What is not here:** the self-employed's *income* and the flat-rate regime (regime forfettario). Eurostat has no income by employment status; that needs the Ministry of Economy IRPEF tables (`docs/manual_data.md`). The policy dates are shown only as markers. **Descriptive only, never causal**: the share of self-employment moves with many things besides tax rules."""),
    (C, '''from cifresignificative.analysis import selfemp as SE, income_dist as ID
se = SE.load_selfemp()
S = SE.share_wide(se)
WIN = {"2005-09": (2005, 2009), "2015-19": (2015, 2019), "2021-25": (2021, 2025)}
print("rows:", len(se), "| tables:", se.groupby("table").size().to_dict())
for c, lab in [("SELF", "all self-employed"), ("SELF_NS", "own-account (no employees)"), ("SELF_S", "employers")]:
    print(lab, "- % of employment 15-64"); display(SE.window_gap(S, c, WIN).round(2))
top = ID.window_table(S, WIN["2015-19"]).SELF.sort_values(ascending=False).head(6).round(1); print("highest in 2015-19:", top.to_dict())
print("Italy's yearly series (% of employment):"); display(S.xs(ITALY, level="iso3").round(1).T)
print("Change in the own-account share, window means 2012-14 -> 2022-25, Italy vs placebo ranking:", SE.rank_in(SE.change_between(S, "SELF_NS", (2012, 2014), (2022, 2025))))
print("Change, 2005-09 -> 2015-19:", SE.rank_in(SE.change_between(S, "SELF_NS", (2005, 2009), (2015, 2019))), "| 2015-19 -> 2021-25:", SE.rank_in(SE.change_between(S, "SELF_NS", (2015, 2019), (2021, 2025))))'''),
    (C, '''rows = []
for y in (2010, 2019, 2024):
    F = SE.sector_frame(se, y); D = SE.decompose_all(F); d = SE.decompose(F)
    rows.append({"year": y, "countries": len(F), "Italy share %": d["share"], "pooled others %": d["pooled_others"], "gap pp": d["share"] - d["pooled_others"], "composition pp": d["composition"], "within-sector pp": d["within_sector"],
                 "rank within (1 = largest)": SE.rank_in(D.within_sector)["rank_from_top"], "p_abs within": SE.rank_in(D.within_sector)["p_abs"], "rank composition": SE.rank_in(D.composition)["rank_from_top"], "p_abs composition": SE.rank_in(D.composition)["p_abs"]})
display(pd.DataFrame(rows).round(2).set_index("year"))
F = SE.sector_frame(se, 2019); it = F.loc[ITALY]; oth = F.drop(ITALY)
sect = pd.DataFrame({"Italy rate %": it["self"] / it["emp"] * 100, "others rate %": oth["self"].sum() / oth["emp"].sum() * 100, "Italy employment weight %": it["emp"] / it["emp"].sum() * 100, "others weight %": oth["emp"].sum() / oth["emp"].to_numpy().sum() * 100})
sect["contribution to the within-sector gap, pp"] = sect["Italy employment weight %"] / 100 * (sect["Italy rate %"] - sect["others rate %"])
print("2019, by NACE section (countries with complete data: %d)" % len(F)); display(sect.round(1).sort_values("contribution to the within-sector gap, pp", ascending=False))
A = SE.age_wide(se)
for a in ("Y15-24", "Y25-49", "Y50-64"): print("age", a, "-> self-employed % of employment in the band"); display(SE.window_gap(A, a, WIN).round(2))'''),
    (C, '''P = SE.poverty_wide(se)
PW = {"2015-19": (2015, 2019), "2021-25": (2021, 2025)}
for c, lab in [("NSAL", "at-risk-of-poverty rate, employed except employees (self-employed and family workers), %"), ("SAL", "employees, %"), ("gap_nsal_sal", "gap NSAL minus SAL, pp"), ("ratio_nsal_sal", "ratio NSAL / SAL")]:
    print(lab); display(SE.window_gap(P, c, PW).round(2))
print("countries with the rate:", P.NSAL.groupby(level="iso3").count().gt(0).sum(), "| first year:", P.NSAL.dropna().index.get_level_values("year").min())'''),
    (C, '''pol = pd.read_csv("data/external/italy_policies.csv", parse_dates=["date"]); fr = pol[pol.title.str.contains("forfettario", case=False)][["date", "title", "reference", "evidence"]]
display(fr)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axes[0]; z = S.SELF_NS.unstack("iso3")
for c in z: ax.plot(z.index, z[c], color="#d0d0cc", lw=0.7)
ax.plot(z.index, z.drop(columns=ITALY).median(axis=1), color="#eda100", ls="--", lw=1.6, label="median of the others"); ax.plot(z.index, z[ITALY], color=ITALY_C, lw=2, label="Italy, own-account")
zs = S.SELF_S.unstack("iso3"); ax.plot(zs.index, zs[ITALY], color=ITALY_C, lw=1.2, ls=":", label="Italy, employers")
ax.axvline(2015, color=INK2, lw=0.6, ls=":"); ax.axvline(2019, color=INK2, lw=0.6, ls=":"); ax.text(2015.1, ax.get_ylim()[1] * 0.97, "2015: regime created", fontsize=6, va="top"); ax.text(2019.1, ax.get_ylim()[1] * 0.9, "2019: EUR 65k", fontsize=6, va="top")
ax.set_title("Self-employed, % of employment 15-64"); ax.legend(fontsize=7)
ax = axes[1]; d = pd.DataFrame(rows).set_index("year")
ax.bar(range(len(d)), d["composition pp"], color="#9ec5f0", label="sector mix"); ax.bar(range(len(d)), d["within-sector pp"], bottom=d["composition pp"], color=ITALY_C, label="higher rates within sectors")
ax.set_xticks(range(len(d)), d.index); ax.set_ylabel("pp of employment"); ax.set_title("Italy's gap to the pooled others"); ax.legend(fontsize=7)
ax = axes[2]; T = ID.window_table(P, PW["2015-19"]).dropna(subset=["NSAL", "SAL"])
for c in T.index: ax.scatter(T.loc[c, "SAL"], T.loc[c, "NSAL"], color=ITALY_C if c == ITALY else "#8a8a85", s=44 if c == ITALY else 16, zorder=3 if c == ITALY else 2); ax.annotate(c, (T.loc[c, "SAL"], T.loc[c, "NSAL"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
lim = [0, max(T.NSAL.max(), T.SAL.max()) + 2]; ax.plot(lim, lim, color=INK2, lw=0.7, ls=":"); ax.set_xlabel("at-risk-of-poverty rate, employees, 2015-19 (%)"); ax.set_ylabel("employed except employees (%)"); ax.set_title("Poverty by activity status")
fig.tight_layout(); save(fig, "31_self_employment")'''),
]
NOTEBOOKS["31_self_employment"].append((M, """## Takeaways (EU labour force survey and EU-SILC via Eurostat, 2005-2025; descriptive, cross-country)
- **Italy has the second-highest self-employment share in the EU27, after Greece.** Self-employed persons were 23.1% of employment aged 15-64 in 2005-09, 21.2% in 2015-19 (Greece 29.2%, Poland 17.6%, Romania 16.2%) and 19.4% in 2021-25, against medians of the others of 12.6%, 11.9% and 11.6%: a gap of 10.6, 9.2 and 7.8 pp, 2nd of 27 in every window (`p_abs` 0.07, the second-smallest value possible). The gap is not mainly employers: own-account workers without employees are 16.5%, 15.3% and 13.1% (others 8.9%, 8.4%, 7.9%; 3rd, 2nd, 3rd), employers 6.7%, 5.9% and 6.3% (others 4.0%, 3.9%, 3.6%). Italy's yearly series lacks 2005, 2018 and 2021 because those values carry a break flag and are dropped.
- **It is shrinking, at a pace that is ordinary.** From 2012-14 to 2022-25 the own-account share fell 2.9 pp (16.0% in 2014 to 12.6% in 2025) against a median change of -0.2 pp for the others; Italy is 22nd of 27 from the top (`p_abs` 0.22), with Romania (-7.7), Greece (-6.0), Croatia (-4.9) and Cyprus (-3.4) falling more. Windows 2005-09 -> 2015-19 (-1.2 pp, rank 22, `p_abs` 0.48) and 2015-19 -> 2021-25 (-2.1 pp, rank 24, `p_abs` 0.19) are no different. No break in the series stands out around 2015 (regime created) or 2019 (threshold raised to EUR 65,000) in the aggregate share; the aggregate cannot show behaviour just below a threshold, and the survey counts people, not tax regimes.
- **Within sectors, not sector mix.** Against the pooled other countries, Italy's 2019 gap of 8.9 pp splits into 0.5 pp from the sector mix (the weights; Italy's composition ranks 10th of 21, `p_abs` 0.67) and 8.3 pp from higher self-employment rates inside the same sectors (2nd of 21, `p_abs` 0.10). The same holds in 2010 (9.4 of 9.9 pp within) and 2024 (6.8 of 7.4). The largest contributions to the within-sector part come from wholesale and retail trade (2.6 pp; 31.6% of workers self-employed against 13.7%), professional and scientific activities (2.1 pp; 57.0% against 26.0%), construction (0.9 pp), manufacturing (0.8 pp), health and social work, and accommodation and food (0.6 pp each); agriculture goes the other way (-0.6 pp: 37.6% against 52.4%). Only the 21 countries with complete sector tables enter (Eurostat leaves out small cells, which are set to zero when the totals add up to at least 97%).
- **At every age band.** The self-employed share is higher than the median of the others by 6.6 pp at 15-24 (10.2% against 3.6%, 2015-19), 9.2 pp at 25-49 and 7.7 pp at 50-64 (2nd of 27 in each); the gap has narrowed in the youngest band (5.1 pp in 2021-25) and at 25-49 (6.7 pp), not at 50-64 (8.1 pp).
- **Poverty: the self-employed are not unusually poor *relative to other countries*, the employees are.** At-risk-of-poverty rate (60% of the median, 18-64) of employed persons other than employees (mostly self-employed): 18.7% in 2015-19 and 17.6% in 2021-25, close to the median of the others (17.7% and 15.8%; ranks 13th and 10th of 27, `p_abs` 0.96 and 0.67). Employees are the unusual group: 10.1% against 6.1% (3rd highest of 27, `p_abs` 0.15) and 8.9% against 6.3% in 2021-25 (4th). The self-employed-to-employee ratio is 1.9 and 2.0, below the median of the others (3.0 and 2.9; 23rd of 27 both times) and the gap 8.7 pp against 12.0 and 9.9, so in Italy the self-employed are about as poor as elsewhere but employees are poorer than elsewhere. Several countries have more contributing family workers in this category, so it is not a clean self-employed rate.
- **Limits.** Counts of people, not income: the survey says nothing on declared income or the flat-rate regime, which needs the MEF IRPEF tables (manual download; open). Break flags remove some Italian years. The poverty comparison uses an activity-status definition (`NSAL`) that includes family workers and the previous year's main activity; sample sizes of self-employed groups are small in several countries. The 2015 date of the regime's creation (L. 190/2014) is taken from the reference in `italy_policies.csv`, not separately verified. Everything is descriptive: shares move with sector structure, tax rules, labour regulation and the cycle together.
"""))

NOTEBOOKS["32_self_employed_income"] = [
    (M, """# Declared income of the self-employed and the flat-rate regime (MEF IRPEF return statistics)
Notebook 31 counted the self-employed; this one looks at **declared income** with the Ministry of Economy and Finance (MEF) personal income tax statistics by income class, tax years 2008-2024 (files downloaded from the Department of Finance open-data portal, `data/external/source/mef_irpef/`; loader `src/cifresignificative/sources/mef_irpef.py`, analysis `src/cifresignificative/analysis/irpef.py`, table `data/processed/mef_irpef.parquet`).
**How to read the tables (this decides what can be said).** (1) The prefix of each file is the *filing* year; income refers to the *tax year = filing year - 1*, used throughout. (2) Taxpayers are classed by **reddito complessivo**, the ordinary IRPEF income. Income taxed under the flat-rate regimes (regime forfetario, regime di vantaggio, the older minimi) is taxed separately and is **not** in it, so a flat-rate filer without other income sits in the zero class whatever the size of the business: by-class distributions describe the *ordinary-regime* income, the flat-rate income exists only as counts and totals. (3) Amounts are thousand euro in the source, euro here. (4) Classes below zero are left out of means: tax year 2017 has an anomaly there and 2018 a definition change (see the checks).
**So this notebook cannot test whether declared income piles up just under the EUR 65,000 revenue threshold**: the threshold is on revenue, the classes are on a different income, and the flat-rate income is not classed. It can describe how many filers the regime has, their average net income, and how the population of ordinary-regime professionals changed. **Descriptive only, never causal.**"""),
    (C, '''from cifresignificative.analysis import irpef as IR
from cifresignificative.sources import mef_irpef as MI
d = pd.read_parquet(PROCESSED / "mef_irpef.parquet")
print("rows:", len(d), "| tables:", d.code.nunique(), "| tax years:", d.tax_year.min(), "-", d.tax_year.max(), "| filing = tax year + 1:", bool((d.tax_year == d.filing_year - 1).all()))
cov = d.groupby("code").tax_year.agg(first="min", last="max", years="nunique"); print("coverage by table (the server returned empty files for the other years):"); display(cov)
q = MI.qa(d); ok = q[(q.suppressed == 0) & q.total.notna()]
print("class rows vs TOTALE row, cells without suppression:", len(ok), "| max |ratio-1|:", round(float((ok.class_sum_ratio - 1).abs().max()), 5), "| suppressed cells in the data:", round(float(d.suppressed.mean()) * 100, 2), "%")
p = d[d.measure.isin(["freq", "amount_keur", "mean_keur"]) & ~d.is_total].pivot_table(index=["code", "tax_year", "class_label", "variable"], columns="measure", values="value", observed=True)
p = p[(p.freq >= 1000) & p.amount_keur.notna() & p.mean_keur.notna()]; print("reported mean = amount / taxpayers (to rounding) in", round(float(((p.mean_keur - p.amount_keur / p.freq).abs() <= 0.0051).mean()) * 100, 2), "% of", len(p), "cells")
print("cross-check, taxpayers in the distribution table = taxpayers in the multi-variable table: identical in all", int(d[(d.variable == "taxpayers") & (d.measure == "freq") & (d.code != "ABPFPRO0303")].groupby(["group", "tax_year", "class_label"]).value.nunique().max() == 1) and "table pairs")
print("jump-flagged amount cells:", int(d.jump.sum()), "| by tax year:", d[d.jump].groupby("tax_year").size().to_dict())
lo = d[(d.code == "AAPFTOT0106") & (d.class_label == "minore di -1.000") & (d.measure == "freq") & (d.variable == "taxpayers")].set_index("tax_year").value
print("taxpayers with reddito complessivo below -1,000 euro:", lo.astype(int).to_dict())
b = d[(d.code == "ABPFTIT0203") & (d.variable == "Reddito d'impresa dichiarato") & (d.measure == "amount_keur") & (d.tax_year.isin([2016, 2017, 2018]))]
print("business income of VAT-number holders, EUR million, TOTALE row vs classes from zero up:"); display(pd.DataFrame({"TOTALE row": b[b.is_total].set_index("tax_year").value / 1000, "classes >= 0": b[~b.is_total & (b.class_lo >= 0)].groupby("tax_year").value.sum() / 1000}).round(0))'''),
    (C, '''f = IR.flat_rate(d)
show = pd.DataFrame({"VAT-number holders (m)": f.tit_taxpayers / 1e6, "forfettario filers (m)": f.get("forfetario_filers") / 1e6, "forfettario net income (EUR bn)": f.get("forfetario_net_income") / 1e9, "forfettario mean net income (EUR)": f.get("forfetario_mean"),
                     "vantaggio filers (k)": f.get("vantaggio_filers") / 1e3, "minimi filers (k)": f.get("minimo_filers") / 1e3, "share of VAT-number holders in a flat regime": f.flat_share_of_tit,
                     "forfettari with ordinary income >= 1,000 EUR (share)": f.get("forfetario_with_other_income") / f.get("forfetario_filers")})
display(show.round(3).T)
mi = pd.DataFrame({g: IR.mean_income(d, g) for g in ["TOT", "DIP", "PEN", "TIT", "IMP", "PRO", "AGR"]}); med = pd.DataFrame({g: IR.median_income(d, g) for g in ["TOT", "DIP", "PEN", "TIT", "IMP", "PRO"]})
print("mean reddito complessivo per taxpayer (EUR, classes from zero up):"); display(mi.round(0).T)
print("median reddito complessivo (EUR, interpolated within the class):"); display(med.round(0).T)
tw = IR.mean_with_flat(d); print("VAT-number holders: mean reddito complessivo + flat-regime net income, per taxpayer (EUR):"); display(pd.DataFrame({"reddito complessivo only": mi.TIT, "plus flat-regime net income": tw, "employees (DIP) mean": mi.DIP, "ratio, VAT-number holders / employees": tw / mi.DIP}).round(2).T)
print("taxpayers, millions:"); display(pd.DataFrame({g: IR.taxpayers(d, g) / 1e6 for g in ["DIP", "PEN", "TIT", "IMP", "PRO", "AGR"]}).round(2).T)
print("professionals (ordinary regime): mean income from self-employment (EUR) and share with reddito complessivo >= 35,000 EUR:"); display(pd.DataFrame({"mean reddito da lavoro autonomo": IR.mean_income(d, "PRO", "Reddito da lavoro autonomo"), "share >= 35k, professionals": IR.share_at_least(d, "PRO", 35000), "share >= 35k, entrepreneurs": IR.share_at_least(d, "IMP", 35000), "share >= 35k, employees": IR.share_at_least(d, "DIP", 35000)}).round(3).T)'''),
    (C, '''fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axes[0]; ys = f.index
ax.bar(ys - 0.0, (f.get("forfetario_filers").fillna(0) / 1e6), color="#2a78d6", label="forfettario"); ax.bar(ys, f.get("vantaggio_filers").fillna(0) / 1e6, bottom=f.get("forfetario_filers").fillna(0) / 1e6, color="#9ec5f0", label="vantaggio")
ax.bar(ys, f.get("minimo_filers").fillna(0) / 1e6, color="#eda100", label="minimi (to 2014)"); ax2 = ax.twinx(); ax2.plot(ys, f.flat_share_of_tit * 100, color=INK2, marker="o", ms=3, lw=1.2); ax2.set_ylabel("% of VAT-number holders (line)"); ax2.grid(False)
ax.set_ylabel("filers, millions"); ax.set_title("Flat-rate regimes"); ax.legend(fontsize=7, loc="upper left")
ax = axes[1]
for g, c, ls in [("DIP", "#2a78d6", "-"), ("PEN", "#8a8a85", "-"), ("PRO", "#d94f3d", "-"), ("IMP", "#eda100", "-")]: ax.plot(mi.index, mi[g] / 1000, color=c, ls=ls, lw=1.8, label={"DIP": "employees", "PEN": "pensioners", "PRO": "professionals (ordinary)", "IMP": "entrepreneurs (ordinary)"}[g])
ax.plot(tw.index, tw / 1000, color="#2a78d6", ls="--", lw=1.4, label="all VAT-number holders + flat-regime income"); ax.set_ylabel("EUR thousand per taxpayer"); ax.set_title("Mean declared income (reddito complessivo)"); ax.legend(fontsize=6)
ax = axes[2]; n = IR.taxpayers(d, "PRO") / 1e6; ax.bar(n.index, n.values, color="#9ec5f0"); ax.set_ylabel("professionals, millions (bars)"); ax3 = ax.twinx(); ax3.plot(mi.index, mi.PRO / 1000, color="#d94f3d", marker="o", ms=3); ax3.set_ylabel("mean income, EUR thousand (line)"); ax3.grid(False); ax.set_title("Professionals: fewer, and richer on average")
for a in (axes[0], axes[1], axes[2]): a.axvline(2018.5, color=INK2, lw=0.6, ls=":")
fig.tight_layout(); save(fig, "32_self_employed_income")'''),
]
NOTEBOOKS["32_self_employed_income"] += [
    (M, """## Costs of the professionals who stayed in the ordinary regime, by turnover band (AE tables)
New data (downloaded 2026-10-15, `data/external/source/mef_irpef_fatturato/`): the MEF tables of individuals **by prevalent income** classed by **VAT turnover**, with the items of the business-income form sections (`AE*`, `sources/mef_irpef.py`; the table codes changed in 2014 and the file for tax year 2022 is empty on the server). The regular professionals' table (form section RE) gives total fees and total expenses per turnover class, so the **cost share** (expenses / fees) can be compared with the flat-rate regime's deemed costs: for professional activities the regime taxes 78% of revenue, i.e. it assumes costs of 22% (coefficient quoted from the regime's rules, to be checked against the Agenzia delle Entrate).
**What this is and is not.** Flat-rate filers use another form section and are *not* in these tables, so this is the group that **stayed** in the ordinary regime, a selected group: if low-cost professionals move to the flat regime, the cost share of those who remain rises. The classes are on turnover (a VAT concept), coarse around the EUR 65,000 (2019-22) and 85,000 (from 2023) thresholds. Descriptive only: no bunching test and no causal reading."""),
    (C, '''dims = pd.read_parquet(PROCESSED / "mef_irpef_dims.parquet")
pc = IR.professional_costs(dims)
order = pc.drop_duplicates("band").sort_values("band_order").band.tolist()
cs = pc.pivot(index="band", columns="tax_year", values="cost_share").loc[order]
fl = pc.pivot(index="band", columns="tax_year", values="filers").loc[order]
print("tax years:", [int(y) for y in sorted(pc.tax_year.unique())], "(2022 is empty on the server)")
print("cost share = expenses / fees, regular professionals in the ordinary regime, by turnover band:"); display(cs.round(3))
print("professionals with fees (thousands):"); display((fl / 1e3).round(0))
chg = (cs[2019] - cs[2018]).rename("change 2018 -> 2019"); chg2 = (cs[2024] - cs[2019]).rename("change 2019 -> 2024")
display(pd.concat([chg, chg2, (fl[2019] / fl[2018] - 1).rename("filers 2018 -> 2019"), (fl[2024] / fl[2018] - 1).rename("filers 2018 -> 2024")], axis=1).round(3))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ax = axes[0]
for y, c in [(2013, "#9ec5f0"), (2018, "#2a78d6"), (2019, "#d94f3d"), (2024, "#eda100")]: ax.plot(range(len(order)), cs[y].values * 100, marker="o", ms=4, color=c, lw=1.6, label=f"tax year {y}")
ax.axhline(22, color=INK2, lw=0.8, ls="--"); ax.text(len(order) - 1, 22.6, "22% deemed (flat regime)", ha="right", fontsize=7, color=INK2)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=35, fontsize=7); ax.set_ylabel("expenses as % of fees"); ax.set_title("Cost share by VAT-turnover band"); ax.legend(fontsize=7)
ax = axes[1]
for b, c in [("25-50k", "#2a78d6"), ("50-60k", "#9ec5f0"), ("60-75k", "#eda100"), ("75-100k", "#d94f3d"), ("100-150k", "#8a8a85")]: ax.plot(cs.columns, cs.loc[b].values * 100, marker="o", ms=3, lw=1.5, color=c, label=b)
ax.axhline(22, color=INK2, lw=0.8, ls="--"); ax.axvline(2018.5, color=INK2, lw=0.6, ls=":"); ax.set_ylabel("expenses as % of fees"); ax.set_title("Cost share over time (dotted: flat-rate threshold raised to 65,000 in 2019)"); ax.legend(fontsize=7, title="turnover band")
fig.tight_layout(); save(fig, "32_professional_costs")'''),
]

NOTEBOOKS["32_self_employed_income"].append((M, """## Takeaways (MEF IRPEF return statistics, tax years 2008-2024; descriptive)
- **The data are sound once read correctly.** 238 of the 315 files requested are non-empty (the server returned empty files for filing years 2005-2008, for the business-owner detail tables of 2009-2011 and for the 2016 employee and pensioner detail), giving tax years 2008-2024 for the distribution tables and 2011/2012-2024 for the detail on VAT-number holders. Where nothing is suppressed the class rows add up to the TOTALE row within 0.13%; the reported mean equals amount / taxpayers (to rounding) in 99.88% of 120,514 cells; taxpayer counts are identical in the paired tables; 3.4% of cells are suppressed (`***`). Three encodings, Italian number format and drifting variable names were normalised in the loader. The file prefix is the *filing* year: income is that of the previous year.
- **One anomaly and one definition change, both in the lowest classes.** Taxpayers with reddito complessivo below -1,000 EUR number 118,476 in tax year 2016, 325,658 in 2017 and 3,414 in 2018 (about 2-4 thousand since). In 2017 that class carries a -10.4 bn EUR business-income amount (91 cells flagged, all of tax year 2017), so the TOTALE row of business income is 20.7 bn against 31.1 bn from the classes at zero and above; a plausible cause is the 2017 switch to cash-basis accounting for simplified-regime businesses, but the files do not say. Negative classes are therefore left out of means and counts; other flagged jumps (140 cells in all) are mostly genuine regime changes (flat-rate net income by class in 2019, the 2021 integration treatment).
- **The flat-rate regimes now cover about half of the VAT-number holders.** The share of individuals with a VAT number in a flat regime rose from 10% (2013, the old minimi regime, 0.40 m) to 21% in 2016, 27% in 2018, 43% in 2019 and 48.5% in 2024, when 1.85 m filers declared 31.8 bn EUR of net income under the regime forfettario (mean 17.2 thousand EUR; 9.2 thousand in 2018, 13.9 thousand in 2019). The step is in tax year 2019: forfettario filers 0.77 m to 1.44 m (+0.67 m) while ordinary-regime professionals fell from 0.74 m to 0.47 m and entrepreneurs from 1.51 m to 1.23 m, in the year the revenue threshold was raised to EUR 65,000 (L. 145/2018). That is timing, not a causal estimate: the regime's eligibility rules changed in several ways at once. About 37% of forfettari in 2024 also report at least 1,000 EUR of ordinary income (30% in 2015, 38% in 2019), i.e. many hold another job, pension or rental income.
- **Declared income of the self-employed, depending on what is counted.** Individuals with a VAT number declare on average 36.7 thousand EUR in 2024 once the flat-regime net income is added (28.4 thousand on reddito complessivo alone), against 26.7 thousand for employees: ratios of 1.37 and 1.06, and 1.20 in 2013 (26.9 vs 22.5 thousand). Their *median* reddito complessivo falls from 14.7 thousand (2013) to 9.8 thousand (2024) because flat-regime filers sit near zero: that is a classification effect, not falling income (employees' median rose from 19.3 to 23.1 thousand).
- **The ordinary-regime professionals became fewer and richer, which is mostly selection.** Professionals fell from 0.94 m (2013) to 0.53 m (2024) while their mean income from self-employment rose from 34.9 to 65.5 thousand EUR and the share with reddito complessivo of at least 35,000 EUR from 47% to 81% (employees: 13% to 19.5%; entrepreneurs: 15% to 33%). The largest step coincides with the flat-regime step of 2019 (0.74 m to 0.47 m professionals, mean 45.2 to 57.2 thousand). The data cannot separate selection from growth.
- **Costs of the professionals who stayed in the ordinary regime (AE tables, turnover bands).** Expenses are 23-27% of fees below EUR 150,000 of turnover in 2018, 37% at 150-250 thousand and 45% above 250 thousand; the flat regime assumes 22% for professional activities (rule quoted, unchecked). In 2019, when the flat-regime threshold went to EUR 65,000, the number of ordinary-regime professionals fell by 47% (0-25 thousand), 54% (25-50 thousand) and 49% (50-60 thousand), against 3-7% above 100 thousand; the cost share of those who remained rose by 4.7 pp (25-50 thousand: 22.8% to 27.4%) and 5.1 pp (50-60 thousand) against 0.3-0.7 pp above 100 thousand. That pattern is what selection would produce (low-cost filers leave, high-cost ones stay); it is not a measure of anyone's true costs, and flat-rate filers are not in these tables. Since 2019 the shares have fallen back in every band (25-50 thousand: 27.4% in 2019, 23.1% in 2024; above 250 thousand: 46.0% to 37.1%). Tax year 2022 is empty on the server; the entrepreneurs' tables (sections RF and RG) are loaded but not analysed.
- **What cannot be done with these tables.** (1) No bunching test at the EUR 65,000 threshold: the threshold is on revenue, the classes are on reddito complessivo, flat-regime income is not classed (63% of forfettari sit in the zero and 0-1,000 EUR classes), and the classes are coarse anyway. That test needs revenue-level administrative data (Agenzia delle Entrate). (2) No distribution of flat-regime income by class, only counts, totals and means. (3) No link to individuals' other attributes (sector, age, region, which are available in other portal tables not downloaded here).
- **Limits.** Tax year 2015 lacks the employee and pensioner detail (empty server files), 2009-2011 the VAT-number detail; small cells are suppressed; amounts are declared, not true, income (the tax-gap evidence in notebook 09 applies); class medians are interpolated within wide classes; the flat-regime counts for 2015 (the first year) are partly suppressed in the TOTALE row, so they are summed from the classes.
"""))

NOTEBOOKS["33_irpef_regions_sectors_ages"] = [
    (M, """# The MEF income tax returns by region, sector, age, turnover and income twentile
Notebook 32 used the returns by income class. The portal also publishes them by **region**, **sector of activity**, **age class**, **VAT-turnover class** and **twentile of reddito complessivo** (`ventili`); they were downloaded with `scripts/fetch_mef_irpef.py` (individuals only, tax years 2008-2024 where available) and parsed by `cifresignificative.sources.mef_irpef` into `data/processed/mef_irpef_dims.parquet`; analysis code `src/cifresignificative/analysis/irpef_dims.py`.
Same reading rules as notebook 32: tax year = filing year - 1; taxpayers are ranked or grouped by **reddito complessivo**, which excludes income taxed under the flat-rate regimes (`MIN` below = flat-rate filers: minimi, regime di vantaggio, regimi agevolati); amounts are euro (source: thousand euro). Two traps found while loading: NACE section names contain `;` (they broke the row split, fixed and verified), and tax year 2024 uses the **ATECO 2025** section list (some sections split or renamed; mapped back where one to one).
**Descriptive only.** These are declared incomes of taxpayers who file, not household income and not true income."""),
    (C, '''from cifresignificative.analysis import irpef_dims as ID
from cifresignificative.analysis import irpef as IR
from cifresignificative.sources import mef_irpef as MI
dm = pd.read_parquet(PROCESSED / "mef_irpef_dims.parquet").query("model != 'AE'"); inc = pd.read_parquet(PROCESSED / "mef_irpef.parquet")
print("rows:", len(dm), "| tables per classification:", dm.groupby("dimension").code.nunique().to_dict(), "| tax years:", dm.tax_year.min(), "-", dm.tax_year.max())
q = MI.qa(dm); ok = q[(q.suppressed == 0) & q.total.notna()]
print("class rows vs TOTALE row, cells without suppression:", len(ok), "| cells off by more than 0.5%:", int(((ok.class_sum_ratio - 1).abs() > 0.005).sum()), "| suppressed cells:", round(float(dm.suppressed.mean()) * 100, 2), "%")
p = dm[dm.measure.isin(["freq", "amount_keur", "mean_keur"]) & ~dm.is_total].pivot_table(index=["dimension", "code", "tax_year", "class_label", "variable"], columns="measure", values="value", observed=True)
p = p[(p.freq >= 1000) & p.amount_keur.notna() & p.mean_keur.notna()]; print("reported mean = amount / taxpayers (to rounding) in", round(float(((p.mean_keur - p.amount_keur / p.freq).abs() <= 0.0051).mean()) * 100, 2), "% of", len(p), "cells")
allv = pd.concat([inc, dm]); t = allv[(allv.variable == "taxpayers") & (allv.measure == "freq") & allv.is_total].groupby(["group", "tax_year", "dimension"]).value.max().unstack("dimension")
print("TOTALE taxpayers identical across income-class, region, sector, age and turnover tables (max relative difference):", float(t.sub(t.median(axis=1), axis=0).abs().div(t.median(axis=1), axis=0).max().max()))
print("empty files on the server: regional tables of 2016 for employees and pensioners (2 of 221)")
print("regions: all 20 present, Trentino and Alto Adige summed; regional rows add up to the TOTALE row in every year:", bool(((ID.by_class(dm, "region", "AAPFTOT0201").sum() / ID.by_class(dm, "region", "AAPFTOT0201", total=True).iloc[0]).round(4) == 1).all()))'''),
    (M, "## 1. Regions: declared income and the take-up of the flat-rate regimes"),
    (C, '''am = ID.area_mean(dm, "AAPFTOT0201"); rm = ID.regional_mean(dm, "AAPFTOT0201")
print("mean reddito complessivo per taxpayer, EUR, by macro-area (all individuals who file):"); display(am.round(0)[[2012, 2016, 2019, 2022, 2024]])
print("South / North-West:", (am.loc["South"] / am.loc["North-West"]).round(3)[[2012, 2016, 2019, 2024]].to_dict(), "| Islands / North-West:", (am.loc["Islands"] / am.loc["North-West"]).round(3)[[2012, 2024]].to_dict())
print("regions in 2024, highest and lowest:", rm[2024].round(0).sort_values().iloc[[0, 1, 2, -3, -2, -1]].to_dict())
dip, tit = ID.area_mean(dm, "AAPFDIP0201"), ID.area_mean(dm, "ABPFTIT0201"); print("VAT-number holders / employees, mean reddito complessivo, by macro-area (flat-regime income not included):"); display((tit / dip).round(2)[[2013, 2016, 2019, 2024]])
fs = ID.flat_share(dm, "region", "ABPFTIT0101", "ABPFMIN0101"); fs = fs.reindex([r for r in ID.AREA if r in fs.index])
print("flat-rate filers as a share of VAT-number holders, by region:"); display(fs[[2013, 2016, 2019, 2024]].round(3).sort_values(2024).T)
area = pd.Series(ID.AREA); fa = (ID.by_class(dm, "region", "ABPFMIN0101").reindex(area.index).groupby(area).sum() / ID.by_class(dm, "region", "ABPFTIT0101").reindex(area.index).groupby(area).sum()); print("by macro-area:"); display(fa[[2013, 2016, 2019, 2024]].round(3))'''),
    (M, "## 2. Sectors of activity"),
    (C, '''s = ID.by_sector(dm, "ABPFTIT0105"); m = ID.by_sector(dm, "ABPFMIN0102"); fsec = (m / s)
last = 2023  # tax year 2024 follows the ATECO 2025 list
tab = pd.DataFrame({"VAT-number holders 2023 (k)": s[last] / 1000, "flat-rate share 2013": fsec[2013], "2016": fsec[2016], "2019": fsec[2019], "2023": fsec[last]}).dropna(subset=["2023"])
tab = tab[tab["VAT-number holders 2023 (k)"] >= 5].sort_values("VAT-number holders 2023 (k)", ascending=False)
print("sections with at least 5,000 VAT-number holders (ATECO 2007 names; tax year 2023):"); display(tab.round(3))
print("share of all VAT-number holders in 2023 held by the five largest sections:", round(float(tab["VAT-number holders 2023 (k)"].head(5).sum() / (s[last].sum() / 1000)), 3), tab.index[:5].tolist())
print("total flat-rate share of VAT-number holders, 2013 -> 2019 -> 2023 -> 2024:", {y: round(float(ID.by_class(dm, "activity", "ABPFMIN0102", total=True)[y].iloc[0] / ID.by_class(dm, "activity", "ABPFTIT0105", total=True)[y].iloc[0]), 3) for y in (2013, 2019, 2023, 2024)})'''),
    (M, "## 3. Age"),
    (C, '''ages = {}
for code, g, y0 in [("AAPFDIP0103", "employees", 2012), ("ABPFTIT0103", "VAT-number holders", 2013), ("ABPFPRO0103", "professionals", 2013), ("ABPFIMP0103", "entrepreneurs", 2013), ("AAPFPEN0103", "pensioners", 2012)]:
    a = ID.by_class(dm, "age", code); ages[g] = (a / a.sum() * 100).reindex(["15 - 24", "25 - 44", "45 - 64", "oltre 64"])[[y0, 2019, 2024]]
    print(g); display(ages[g].round(1).T)
n = {g: ID.by_class(dm, "age", c).sum() / 1e6 for c, g in [("ABPFPRO0103", "professionals"), ("ABPFIMP0103", "entrepreneurs")]}; print("taxpayers (m), 2013 / 2019 / 2024:", {g: [round(float(v[y]), 2) for y in (2013, 2019, 2024)] for g, v in n.items()})'''),
    (M, "## 4. VAT turnover: who left the ordinary regime in 2019?\nFlat-rate filers do not declare VAT, so the turnover classes of the VAT-number holders show the **ordinary-regime** population by size, plus a first class (`no VAT declaration`) that holds the flat-rate filers and other VAT-exempt filers."),
    (C, '''tb = ID.turnover_bands(dm, "ABPFTIT0107"); ordb = tb.drop(index="no VAT declaration")
print("VAT-number holders by VAT-turnover band, thousands:"); display((tb / 1000).round(0)[[2013, 2016, 2018, 2019, 2020, 2022, 2024]])
print("share of VAT-number holders with no VAT declaration:", (tb.loc["no VAT declaration"] / tb.sum()).round(3).to_dict())
chg = (ordb[2019] - ordb[2018]).round(0); print("change 2018 -> 2019 in the ordinary-regime population, by band (thousands):", (chg / 1000).round(1).to_dict(), "| total:", round(float(chg.sum()) / 1000, 1))
print("change in the 'no VAT declaration' group 2018 -> 2019 (thousands):", round(float(tb.loc['no VAT declaration', 2019] - tb.loc['no VAT declaration', 2018]) / 1000, 1), "| forfettario filers, notebook 32 table: +", round(float(IR.flat_rate(inc).forfetario_filers.diff()[2019]) / 1000, 1))
print("relative change 2018 -> 2019 by band (%):", ((ordb[2019] / ordb[2018] - 1) * 100).round(1).to_dict())
print("band containing the EUR 65,000 revenue threshold is 60-75k; the pre-2019 threshold was lower and varied by activity (not verified here).")'''),
    (M, "## 5. The top of the distribution: income twentiles"),
    (C, '''sh = ID.ventile_shares(dm)
top = pd.DataFrame({"top 5% (XX ventile)": sh[20], "top 10%": sh[19] + sh[20], "top 25%": sh.loc[:, 16:20].sum(axis=1), "bottom 50% (I-X)": sh.loc[:, 1:10].sum(axis=1), "bottom ventile": sh[1]}) * 100
print("share of total reddito complessivo held by groups of taxpayers (%):"); display(top.round(2).T[[2011, 2012, 2016, 2019, 2020, 2022, 2024]])
b = dm[(dm.dimension == "ventile") & (dm.variable == "ventile_upper_eur") & ~dm.is_total].pivot(index="class_order", columns="tax_year", values="value")
print("income at the upper end of the median ventile (X), 90th (XVIII) percentile, EUR:", b.loc[[9, 17], [2012, 2019, 2024]].round(0).to_dict())
f = dm[(dm.dimension == "ventile") & (dm.variable == "Reddito complessivo") & (dm.measure == "freq") & ~dm.is_total].pivot(index="class_order", columns="tax_year", values="value"); print("taxpayers in the first ventile with an income (m, 2024):", round(float(f.loc[0, 2024]) / 1e6, 2), "vs", round(float(f.loc[1, 2024]) / 1e6, 2), "in the second: zero-income filers sit in the first ventile")
for y in (2012, 2024):
    mix = ID.ventile_mix(dm, y); print(y, "income mix (%) of selected ventiles:"); display((mix.loc[[1, 5, 10, 15, 19, 20]] * 100).round(1))
print("tax year 2017 is distorted in the first ventile (negative business losses in the lowest class, see notebook 32): bottom-half share", round(float(top.loc[2017, 'bottom 50% (I-X)']), 1), "% against", round(float(top.loc[2016, 'bottom 50% (I-X)']), 1), "in 2016")'''),
    (C, '''fig, axes = plt.subplots(2, 3, figsize=(16, 8.4))
ax = axes[0, 0]
for g, c in [("North-West", "#2a78d6"), ("North-East", "#9ec5f0"), ("Centre", "#8a8a85"), ("South", "#eda100"), ("Islands", "#d94f3d")]: ax.plot(am.columns, am.loc[g] / 1000, label=g, color=c, lw=1.8)
ax.set_title("Mean reddito complessivo per taxpayer (EUR thousand)"); ax.legend(fontsize=7)
ax = axes[0, 1]; fa.T.plot(ax=ax, color=["#8a8a85", "#d94f3d", "#eda100", "#2a78d6", "#9ec5f0"], lw=1.8, legend=False); ax.set_title("Flat-rate filers, share of VAT-number holders"); ax.set_xlabel("")
ax = axes[0, 2]; sel = tab.head(8); y = np.arange(len(sel))[::-1]
ax.barh(y, sel["2019"], color="#9ec5f0", label="2019"); ax.scatter(sel["2023"], y, color="#2a78d6", zorder=3, label="2023"); ax.set_yticks(y, [i[:34] for i in sel.index], fontsize=6.5); ax.set_title("Flat-rate share, 8 largest sectors"); ax.legend(fontsize=7)
ax = axes[1, 0]; idx = (ordb.drop(index="<=0").div(ordb.drop(index="<=0")[2018], axis=0) * 100).T
for c, col in zip(idx.columns, ["#9ec5f0", "#2a78d6", "#d94f3d", "#eda100", "#8a8a85"], strict=False): ax.plot(idx.index, idx[c], label=c, color=col, lw=1.6)
ax.axvline(2018.5, color=INK2, lw=0.6, ls=":"); ax.set_title("Ordinary-regime VAT-number holders by turnover band (2018 = 100)"); ax.legend(fontsize=7)
ax = axes[1, 1]
for c, col in [("top 5% (XX ventile)", "#2a78d6"), ("top 10%", "#9ec5f0"), ("bottom 50% (I-X)", "#eda100")]: ax.plot(top.index, top[c], label=c, color=col, lw=1.8)
ax.set_title("Share of reddito complessivo (%)"); ax.legend(fontsize=7)
ax = axes[1, 2]
for g, c in [("employees", "#2a78d6"), ("VAT-number holders", "#eda100"), ("professionals", "#d94f3d"), ("pensioners", "#8a8a85")]: ax.bar(g[:6], ages[g].loc["oltre 64", 2024], color=c)
ax.set_title("Share aged 65+, 2024 (%)")
fig.tight_layout(); save(fig, "33_irpef_dims")'''),
]
NOTEBOOKS["33_irpef_regions_sectors_ages"].append((M, """## Takeaways (MEF IRPEF return statistics, tax years 2008-2024; descriptive)
- **Data checks.** 842 files parse (the first batch of 238 plus 604 downloaded by the new crawler, 2 regional files empty on the server). One bug was found by the checks, not by eye: NACE section names with `;` split rows wrongly (retail trade, water supply and public administration were read as numbers like 7 or 5, so sector rows added up to only 84% of the total); and the section list changes in tax year 2024 (ATECO 2025). After the fix, class rows add up to the TOTALE row in all but one of 14,308 unsuppressed cells (0.7% off, on a tiny cell), the reported mean equals amount / taxpayers in 99.9% of 164,670 cells, and TOTALE taxpayers are identical across the income-class, region, sector, age and turnover tables for every group and year. 3.7% of cells are suppressed.
- **Regions: the North-South gap in declared income is stable.** Mean reddito complessivo per taxpayer in 2024 is 28.3 thousand EUR in the North-West, 26.9 in the North-East, 25.9 in the Centre, 20.3 in the Islands and 20.2 in the South; the South stands at 0.70 of the North-West in 2012 and 0.72 in 2024 (Lombardy 29.4 thousand, Calabria 18.5). VAT-number holders earn about as much as employees on ordinary income in the South and Islands (ratio 1.01 in 2024, 0.97-0.98 in 2013) and 7-17% more in the North (1.07 North-West, 1.17 North-East).
- **The flat-rate regimes spread everywhere, a bit more in the Centre.** The share of VAT-number holders in the flat-rate group rose from 11-15% (2013) to 48-55% (2024): Centre 55%, North-West 54%, Islands 51%, South 50%, North-East 48%; by region from 38% (Trentino-Alto Adige) and 43% (Basilicata) to 60% (Lazio). The regional differences are small next to the national rise. (This group, 52% of VAT-number holders in 2024, is larger than the 48.5% with positive forfettario net income in notebook 32: it also holds filers with zero net income.)
- **Sectors: the regime is a services phenomenon and the largest sectors differ.** Five sections hold 71% of the 3.8 million VAT-number holders in 2023 (professional and technical activities 1.01 million, trade 0.67, health and social work 0.43, construction 0.34, agriculture 0.24). The flat-rate share rose between 2013 and 2023 from 23% to 67% in professional activities, 6% to 37% in trade, 22% to 57% in health, 9% to 42% in construction and 2% to 21% in accommodation and food; agriculture stays at 2% (a separate VAT regime). Tax year 2024 follows the ATECO 2025 list and is left out of the sector comparison.
- **Age: the ordinary-regime professionals have aged.** Professionals aged 25-44 fell from 36% to 15% of the group between 2013 and 2024 and those over 64 rose from 12% to 32% (the group itself shrank from 0.95 to 0.53 million); entrepreneurs aged 25-44 fell from 38% to 23%. Among all VAT-number holders the 65+ share is 15.5% (11.8% in 2013) against 5.5% of employees (2.4%). The tables have no age split for the flat-rate filers, so that younger professionals moved into the flat regime is an inference.
- **Turnover: the ordinary regime lost small filers in 2019.** The number of VAT-number holders with a VAT declaration fell by 499 thousand between tax years 2018 and 2019: -254 thousand (-34%) with turnover up to 30,000 EUR, -177 thousand (-32%) between 30,000 and 60,000, -28 thousand (-17%) in the 60-75 thousand class that contains the EUR 65,000 threshold, and only -13 thousand (-7%) and -16 thousand (-3%) above, while the group without a VAT declaration (flat-rate and other exempt filers) grew by 543 thousand (forfettario filers +667 thousand in notebook 32) and now holds 56% of VAT-number holders. That is the pattern a higher threshold would produce, but it is a coincidence in time and cannot isolate the rule from the 2019 e-invoicing and other changes.
- **Top shares of declared income are stable.** The top 5% of taxpayers held 22.9% of reddito complessivo in 2011, 22.8% in 2019 and 22.5% in 2024 (peak 23.6% in 2022); the top 10% 33.4%, 33.2% and 32.7%; the top 25% 55.8% and 54.5% (2011 and 2024); the bottom half 18.8% and 19.6% (17.2% in 2017, distorted by the lowest-class anomaly of notebook 32). The median taxpayer's income (upper end of the tenth twentile) rose from 15.7 to 21.2 thousand EUR between 2012 and 2024 and the 90th percentile from 35.8 to 44.5. The top twentile's income in 2024 is 45% employment, 18% pensions, 12.5% self-employment, 6.5% business income, 3% buildings and 14% other (partnerships, miscellaneous); the first twentile holds zero and (before 2018) negative incomes and is mostly property income (49% in 2024).
- **Limits.** Declared ordinary income of those who file: it excludes flat-rate and flat-tax income (the 'cedolare secca' on rents, financial income taxed at source), non-filers and household structure, so top shares are understated and cannot be compared with the EU-SILC quintile shares of notebook 30 (equivalised disposable household income). The first twentile and the negative classes are unreliable before 2018. Regional means mix people who work in one region with filing residence. The sector, turnover and age tables have no flat-rate split except the sector and region tables (group `MIN`).
"""))

NOTEBOOKS["34_regional_south"] = [
    (M, """# Why the South: governance, firm structure, migration and public spending across regions
Notebook 25 found that Italy's regional slopes are the steepest of 14-18 countries and sit in the South and Islands, but it could not say why. Four candidate explanations are added here, in **two designs side by side** (code: `src/cifresignificative/analysis/regional_south.py`, `experiments.run_south_all`):
- **Track A, EU-wide** (can be ranked against other countries): the **EQI** (European Quality of Government Index: survey-based perceived quality, impartiality and corruption of public services; waves 2010, 2013, 2017, 2021, 2024; 186 NUTS2 regions, Germany and Belgium are only published at NUTS1 and drop out) and three **structural business statistics** indicators (manufacturing share of business employment, local units per 1,000 inhabitants, wages per person employed; Eurostat, 2008-2020). Regions are demeaned by country, so only differences *within* a country count. Pre-specified family: EQI wave 2017 and its components + the three SBS indicators against six regional gaps (income per head, output per hour, hours per inhabitant, employment rates, unemployment), outcomes averaged over 2015-19; BH control over the family; other waves are replications.
- **Track B, Italy only** (exploratory; there is no placebo country): **internal migration** between regions (Istat registry transfers) and **public spending by region** (Conti Pubblici Territoriali, extended public sector, per inhabitant, by sector and level of government). With 21 regions in one cluster (8 in the South and Islands) the null permutes values across regions, once freely and once **within the three zones** (North, Centre, South and Islands), which asks whether an association survives the North-South split itself.
**Descriptive associations only, never causal**: perceptions of institutions follow prosperity as much as they could precede it, migration follows jobs, and spending is attributed to where it is incurred (pensions and central-government spending included)."""),
    (C, '''from cifresignificative.analysis import corr, regional_corr as rc, regional_south as rs, experiments as ex
reg, extra = load_regional(), load_regional_extra()
res = ex.run_south_all(reg, extra)
R = rc.regional_matrix(pd.concat([reg, extra], ignore_index=True), drop_flag_chars="u")
e = extra[extra.indicator == "nuts2_eqi"]
print("EQI regions per wave (NUTS2 only):", e.groupby("year").geo.nunique().to_dict(), "| countries:", e.country.nunique(), "| Italian regions per wave:", e[e.geo.str.startswith("IT")].groupby("year").geo.nunique().unique().tolist())
W = rs.within_country(rs.wave_table(R, 2017), income_control=False); g = rc.country_of(W.index)
print("regions with data per indicator in the primary window (2017 wave, outcomes 2015-19):"); display(W[[*rs.EU_WIDE, *rs.OUTCOMES]].notna().sum().to_frame("regions").T)
print("countries with at least 5 regions with data (eligible for Italy's placebo ranking):", int((W[["eqi", "employment_rate_20_64"]].dropna().groupby(g).size() >= 5).sum()))
n = res["s1_null"]
print("\\nCalibration of the permutation test: shuffling every column within its country and rerunning the family gives at least one BH discovery in", int((n.discoveries > 0).sum()), "of", len(n), "shuffles; share of tests with p < 0.05:", round(float(n.share_p_below_005.mean()), 3), "(nominal 0.05)")
rng = np.random.default_rng(2); d = W[["eqi", "employment_rate_20_64"]].dropna(); gg = g.reindex(d.index)
hits = [corr._corr(*[corr.shuffle_within(d, gg, rng)[c].to_numpy() for c in d.columns], "spearman")[1] < 0.05 for _ in range(400)]
print("with the ordinary (asymptotic) Spearman p instead, one shuffled test rejects at 5% in", round(float(np.mean(hits)) * 100, 1), "% of 400 shuffles: countries differ in dispersion, which widens the null of the pooled rho; hence permutation p-values below")'''),
    (M, "## Track A. EU-wide: within-country associations, primary wave (EQI 2017, outcomes 2015-19)\nPay per employee against income or output per hour is near-definitional and left out of the family (40 pairs remain of 42). The first table is the plain within-country association; the second also removes each region's fit on (country-demeaned) log income per head from every non-income variable, i.e. what is left beyond income."),
    (C, '''s1 = res["s1_eu_wide"]
for ic, label in [(False, "plain within-country"), (True, "beyond income")]:
    p = s1[s1.primary & (s1.income_control == ic)]
    print(label, ":", len(p), "tests,", int(p.significant.sum()), "BH discoveries (q <= 0.05); leave-one-country-out sign flips among them:", int(p[p.significant].loo_sign_flips.gt(0).sum()))
tab = s1[s1.primary & ~s1.income_control & s1.x.eq("eqi")].set_index("y").loc[rs.OUTCOMES]
tab2 = s1[s1.primary & s1.income_control & s1.x.eq("eqi")].set_index("y").reindex(rs.OUTCOMES)
display(pd.DataFrame({"n": tab.n, "countries": tab.n_groups, "rho": tab.rho, "q": tab.q_value, "loo min |rho|": tab.loo_min_abs_rho, "rho beyond income": tab2.rho, "q beyond income": tab2.q_value}).round(3))
print("EQI (composite) by wave, rho with each outcome (plain within-country):")
display(s1[~s1.income_control & s1.x.eq("eqi")].pivot(index="y", columns="wave", values="rho").loc[rs.OUTCOMES].round(2))
d = s1[~s1.income_control & s1.primary & s1.significant][["x", "y"]]; m = s1[~s1.income_control].merge(d, on=["x", "y"])
print("the", len(d), "primary discoveries in the other waves: same sign", m.groupby("wave").same_sign_as_primary.mean().round(2).to_dict(), "| q <= 0.05 in the wave", m.groupby("wave").significant.mean().round(2).to_dict())
print("components of the EQI, rho with employment rate 20-64 and unemployment (primary wave, plain):")
display(s1[s1.primary & ~s1.income_control & s1.x.str.startswith("eqi") & s1.y.isin(["employment_rate_20_64", "unemployment_rate"])].pivot(index="x", columns="y", values="rho").round(2))
print("structural business statistics, primary wave: rho plain / beyond income")
a = s1[s1.primary & ~s1.income_control & s1.x.str.startswith("sbs")].set_index(["x", "y"]).rho; b = s1[s1.primary & s1.income_control & s1.x.str.startswith("sbs")].set_index(["x", "y"]).rho
display(pd.DataFrame({"plain": a, "beyond income": b}).unstack("y").round(2))'''),
    (M, "## Is Italy different? Italy's slope against the other countries' (placebo ranking)\nFor each pair, the country slope minus the slope of all the other countries is computed for every country with at least 5 regions with data; Italy's rank among them is its p-value, so the smallest possible p is 1 / number of countries (about 0.06)."),
    (C, '''s2 = res["s2_italy_rank"]
print("countries in the ranking:", int(s2.n_countries.min()), "-", int(s2.n_countries.max()), "| smallest possible p:", round(float((1 / s2.n_countries).min()), 3))
print("pairs where Italy's slope difference is the largest of all countries (p = 1/n):", int((s2.p_diff <= 1 / s2.n_countries + 1e-9).sum()), "of", len(s2))
display(s2.sort_values("p_diff").head(8)[["x", "y", "n_countries", "slope_italy", "slope_others", "p_diff", "slope_north", "slope_south"]].round(3))
print("EQI pairs: Italy's rank p-values", s2[s2.x.str.startswith("eqi")].p_diff.round(2).describe()[["min", "50%", "max"]].to_dict())'''),
    (M, "## Track B. Italy only (exploratory): internal migration and public spending by region\nWindow 2015-19, nominal euro, per inhabitant of the year. `rho` = Spearman across the 21 regions; `p_perm` shuffles values across all regions; `rho_within_zone` and `p_perm_within_zone` demean by zone (North 9 regions, Centre 4, South and Islands 8) and shuffle within zones. q is BH over all pairs of Track B: exploratory (one cluster, indicators chosen by eye)."),
    (C, '''T = res["s_italy_table"].set_index("geo"); s3 = res["s3_italy_only"]
zone = pd.Series({g: rs.zone_of(g) for g in T.index})
print("regions per zone:", zone.value_counts().to_dict(), "| pairs:", len(s3), "| q <= 0.05 unrestricted:", int((s3.q_perm <= 0.05).sum()), "| q <= 0.05 within zones:", int((s3.q_perm_within_zone <= 0.05).sum()))
print("North-South gaps (South and Islands minus North, in standard deviations across the 21 regions):")
gp = res["s5_zone_gaps"].set_index("indicator"); display(gp.loc[[c for c in [*rs.OUTCOMES, *rs.EU_WIDE, "net_migration_per_1000", "outflow_per_1000", "spend_total_pc_eur", "spend_capital_pc_eur", "spend_pensions_pc_eur", "spend_central_level_pc_eur", "spend_regional_level_pc_eur", "spend_local_level_pc_eur", "spend_education_pc_eur", "spend_health_pc_eur"] if c in gp.index]].round(2))
print("migration, per 1,000 inhabitants, 2015-19 averages:"); display(T[["net_migration_per_1000", "inflow_per_1000", "outflow_per_1000", "moves_to_north_per_1000"]].groupby(zone).mean().round(2))
mi = s3[s3.x.isin(["net_migration_per_1000", "inflow_per_1000", "outflow_per_1000", "moves_to_north_per_1000"]) & s3.y.isin(["employment_rate_20_64", "unemployment_rate", "gdp_pps_pc_eu27"])]
display(mi[["x", "y", "rho", "p_perm", "rho_within_zone", "p_perm_within_zone", "loo_min_rho", "loo_max_rho"]].round(3))
sp = s3[(s3.x.str.startswith("spend") | s3.x.eq("capital_share_pct")) & s3.y.isin(["employment_rate_20_64", "gdp_pps_pc_eu27"])]
print("spending (per inhabitant) against employment rate and income per head:")
pv = sp.pivot(index="x", columns="y", values=["rho", "rho_within_zone"]); pv.columns = [f"{a} ({b})" for a, b in pv.columns]
display(pv.join(sp[sp.y == "employment_rate_20_64"].set_index("x")[["p_perm_within_zone"]].rename(columns={"p_perm_within_zone": "p within zones (employment)"})).round(3))
print("pairs surviving within zones (BH q <= 0.05), by x:"); display(s3[s3.q_perm_within_zone <= 0.05].groupby("x").agg(pairs=("y", "size"), rho_within_zone_min=("rho_within_zone", "min"), rho_within_zone_max=("rho_within_zone", "max"), sign_flips=("loo_sign_flips", "max")).round(2))'''),
    (M, "## The two designs side by side\nFor the indicators that exist EU-wide, Italy's own rho is the same number in both designs (a country-demeaned Spearman within one country is the raw Spearman); what differs is the reference: Track A ranks Italy's slope among other countries, Track B asks whether the 21 regions could produce it by chance, with and without the North-South split held fixed."),
    (C, '''s4 = res["s4_compare"]
c = s4[s4.x.isin(["eqi", "eqi_impartiality", "sbs_local_units_per_1000_inhabitants", "sbs_manufacturing_employment_share_pct"]) & s4.y.isin(["employment_rate_20_64", "unemployment_rate", "hours_per_inhabitant"])]
display(c[["x", "y", "n_countries", "slope_italy_A", "slope_others_A", "p_rank_among_countries", "rho_B", "p_perm_B", "rho_within_zone_B", "p_perm_within_zone_B"]].round(3))
print("pairs with an unrestricted Italian permutation p < 0.05:", int((s4.p_perm_B < 0.05).sum()), "| also p < 0.05 within zones:", int(((s4.p_perm_B < 0.05) & (s4.p_perm_within_zone_B < 0.05)).sum()), "| Italy's slope the largest of all countries:", int((s4.p_rank_among_countries <= 1 / s4.n_countries + 1e-9).sum()), "of", len(s4))'''),
    (C, '''fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
ZC = {"North": "#2a78d6", "Centre": "#8a8a85", "South and Islands": "#d94f3d"}
ax = axes[0]; d = W[["eqi", "employment_rate_20_64"]].dropna(); it = d.index.str.startswith("IT")
ax.scatter(d.eqi[~it], d.employment_rate_20_64[~it], s=10, color="#c9c8c3", label="other countries")
for z, col in ZC.items():
    k = d.index.isin([g for g in d.index if g.startswith("IT") and rs.zone_of(g) == z]); ax.scatter(d.eqi[k], d.employment_rate_20_64[k], s=26, color=col, label="Italy: " + z)
ax.axhline(0, color=INK2, lw=0.5); ax.axvline(0, color=INK2, lw=0.5); ax.set_xlabel("EQI 2017, minus country mean"); ax.set_ylabel("employment rate 20-64, minus country mean (pp)"); ax.set_title("Within-country: governance and employment"); ax.legend(fontsize=7)
ax = axes[1]; sel = ["net_migration_per_1000", "spend_total_pc_eur", "spend_capital_pc_eur", "spend_central_level_pc_eur", "spend_local_level_pc_eur", "spend_pensions_pc_eur", "spend_education_pc_eur", "spend_health_pc_eur", "eqi", "sbs_local_units_per_1000_inhabitants"]
q = s3[s3.y.eq("employment_rate_20_64") & s3.x.isin(sel)].set_index("x").loc[sel][::-1]; y = np.arange(len(q))
ax.scatter(q.rho, y, color="#9ec5f0", s=40, label="across 21 regions", zorder=3); ax.scatter(q.rho_within_zone, y, color="#d94f3d", s=40, marker="D", label="within zones", zorder=3)
for yy, a, b in zip(y, q.rho, q.rho_within_zone, strict=True): ax.plot([a, b], [yy, yy], color="#c9c8c3", lw=1)
ax.set_yticks(y, [i.replace("_pc_eur", "").replace("_per_1000", "").replace("spend_", "spend: ")[:30] for i in q.index], fontsize=7.5); ax.axvline(0, color=INK2, lw=0.5); ax.set_xlabel("Spearman rho with employment rate 20-64"); ax.set_title("Italy only: the North-South split vs within zones"); ax.legend(fontsize=7)
ax = axes[2]; gg = gp.loc[[c for c in ["employment_rate_20_64", "gdp_per_hour_eur", "eqi", "sbs_local_units_per_1000_inhabitants", "sbs_wages_per_employee_keur", "sbs_manufacturing_employment_share_pct", "net_migration_per_1000", "spend_total_pc_eur", "spend_capital_pc_eur", "spend_pensions_pc_eur", "spend_education_pc_eur", "spend_health_pc_eur"] if c in gp.index]][::-1]
ax.barh(np.arange(len(gg)), gg.gap_sd, color=["#d94f3d" if v < 0 else "#2a78d6" for v in gg.gap_sd]); ax.set_yticks(np.arange(len(gg)), [i.replace("_pc_eur", "").replace("_per_1000", "").replace("_inhabitants", "")[:34] for i in gg.index], fontsize=7.5); ax.axvline(0, color=INK2, lw=0.5)
ax.set_xlabel("South and Islands minus North, standard deviations"); ax.set_title("How far apart the two ends are")
fig.tight_layout(); save(fig, "34_regional_south")'''),
]
NOTEBOOKS["34_regional_south"].append((M, """## Track B, addition: OpenCoesione 2014-2020 by region
`sources/opencoesione.py` reads the five 2014-2020 project files of OpenCoesione (ERDF, ESF, the national Development and Cohesion Fund FSC, the Complementary Action Plan PAC and ordinary resources; 944 thousand projects, 194 billion euro of public financing net of economies allocated to a region; a further 5% has no Italian localisation, and 8% is in projects of several regions and is divided equally among them). Predictors per inhabitant (mean population 2014-2019): the EU part of all funds, the national co-financing of the EU programmes, the FSC, and the share of the financing paid out by mid-2026. **Pre-specified family**: 4 predictors x the 6 outcomes of this notebook (2015-19), Spearman across the 21 regions, permutation p across regions and within the three zones, BH. Since funds are allocated by rule to the poorer regions, the unrestricted column mostly measures the North-South split; the within-zone column is the one to read. A null with the predictors shuffled across regions must give about zero discoveries.
""")); NOTEBOOKS["34_regional_south"].append((C, '''ores = ex.run_opencoesione_all()
oa, oz, of, on, ou = ores["o1_allocation"].set_index("geo"), ores["o2_zones"].set_index("zone"), ores["o3_family"], ores["o4_null"].round(3).to_dict("records")[0], ores["o5_unallocated"]
print("allocation of the 2014-2020 cycle by zone (EUR per inhabitant; payment rate in %):"); display(oz.round(1))
print("what could not be allocated, and multi-region projects, by fund (EUR million):"); display(ou.round(0))
print("family:", len(of), "tests | BH discoveries unrestricted:", int((of.q_perm <= 0.05).sum()), "| within zones:", int((of.q_perm_within_zone <= 0.05).sum()), "| leave-one-out sign flips among the unrestricted discoveries:", int(of[of.q_perm <= 0.05].loo_sign_flips.gt(0).sum()))
print("null (predictors shuffled across regions):", on)
display(of.pivot(index="x", columns="y", values="rho").round(2)); print("within zones:"); display(of.pivot(index="x", columns="y", values="rho_within_zone").round(2)); print("within-zone permutation p, smallest three:"); display(of.sort_values("p_perm_within_zone").head(3)[["x", "y", "rho_within_zone", "p_perm_within_zone", "q_perm_within_zone"]].round(3))
print("regions with the most and the least EU funds per inhabitant:"); display(oa.sort_values("oc_eu_pc_eur").iloc[[0, 1, 2, -3, -2, -1]].round(1))
'''))
NOTEBOOKS["34_regional_south"].append((M, """## Takeaways (EQI 2010-2024, Eurostat structural business statistics 2008-2020, Istat internal migration, CPT spending; primary window 2015-19; descriptive associations)
- **The permutation test was needed.** The ordinary Spearman p-value rejects a *shuffled* within-country test 8.2% of the time at the nominal 5% (400 shuffles), because countries differ in dispersion. All EU-wide p-values here are within-country permutation p-values (2,000 shuffles, so the smallest p is 0.0005); shuffling every column within its country and rerunning the family gives a BH discovery in 2 of 100 shuffles and p < 0.05 in 5.6% of tests. (The earlier regional screen, notebook 25, uses the asymptotic p with robustness filters; this check was not repeated there.)
- **Track A. Within a country, regions with better perceived governance have more people at work, beyond what income explains.** EQI (2017) against the employment rate 20-64: rho 0.36 across 181 regions in 18 countries; against unemployment -0.41; income per head 0.31; hours worked per inhabitant 0.30; output per hour only 0.19 (q 0.016, and its leave-one-country-out minimum is 0.03). Employment and unemployment hold in all four waves (employment rho 0.34, 0.40, 0.36, 0.40 for 2010-2021; unemployment -0.36 to -0.45). Of 40 pairs, 37 pass BH at 5% in the primary wave, none flips sign when a country is dropped, and the 37 keep their sign in every other wave (q <= 0.05 in 76%, 84% and 92% of them in 2010, 2013 and 2021). After removing each region's fit on income per head, 18 of 40 survive: EQI keeps employment (0.45), female employment (0.44) and unemployment (-0.475, all q 0.001) but its link with income per head (0.02), output per hour (-0.07) and hours per inhabitant (0.08) disappears. All three components point the same way; quality is the strongest (employment 0.36, unemployment -0.42), then corruption (0.33) and impartiality (0.29).
- **Firm structure mostly tracks income.** Local units per 1,000 inhabitants goes with employment (0.47), hours per inhabitant (0.62) and income per head (0.59), but beyond income the link is about zero (employment -0.03). Manufacturing's share of business employment is weakly linked to employment at first (0.07) and to output per hour negatively (-0.17), but beyond income it goes with higher employment (0.39) and lower unemployment (-0.47). Pay per person employed goes with employment (0.50) and unemployment (-0.45) but is near zero beyond income (employment 0.11); pay against income or output per hour is near-definitional and not tested.
- **Italy against the other countries (placebo ranking, 14-17 countries).** In 6 of 40 pairs Italy's slope difference is the largest of all countries, the smallest p the ranking can give (1/n, 0.059 to 0.071, so no pair can reach 0.05): the six are local units per 1,000 inhabitants (employment, female employment, unemployment), the manufacturing share (employment, female employment) and pay per person employed (employment). Example: one more local unit per 1,000 inhabitants goes with 1.04 pp more employment in Italy's regions (1.03 in the South, 0.29 in the North) against 0.09 pp in the other countries. For the EQI Italy is not unusual: its rank p-values are 0.43 to 1.00 (median 0.64), so the governance-employment link looks like the other countries' link, only on a larger spread.
- **Track B. Migration is the North-South split.** South and Islands lose 2.7 residents per 1,000 a year net to other regions (North +1.7): inflow 4.6 against 7.3, outflow 7.3 against 5.6. Across the 21 regions net migration goes with the employment rate (rho 0.90, p < 0.001), income per head (0.87) and unemployment (-0.91), but within the three zones it falls to 0.41 (p 0.10), 0.38 (0.09) and -0.43 (0.09): none of the migration pairs survives BH within zones. Moves to the North per 1,000 inhabitants are weakly related to the labour market within zones (employment -0.46, p 0.052; unemployment 0.48, p 0.040; not BH-significant). Migration follows jobs as much as it could cause the gap.
- **Track B. Spending: the central part follows the split, the capital and local part survives it.** Total spending per inhabitant is 20.2 thousand EUR in the North and 14.9 in the South and Islands (gap 1.4 SD); it goes with income per head (rho 0.82) and employment (0.71), and within zones with income (0.61, p 0.01) and employment (0.36, p 0.07). Central-government spending per inhabitant (rho 0.67 with employment) falls to 0.01 within zones (p 0.95) and pensions (0.71) to 0.28 (p 0.25): they sit where the older, richer population is. Capital spending per inhabitant keeps its association within zones (0.53 across regions, 0.49 within, p 0.01), as does spending by local governments (0.49 and 0.62, p < 0.01); the capital share of spending goes from 0.20 to 0.60 within zones (p < 0.01), and education spending from -0.04 to 0.41 (p 0.03).
- **The designs side by side.** 36 of the 40 EU-wide pairs have an Italian permutation p < 0.05 across the 21 regions and 14 also within zones, among them all four pairs each of EQI impartiality and of local units per 1,000 inhabitants (p <= 0.003 within zones) and the EQI composite with unemployment (-0.61, p 0.004). The manufacturing share does not survive the zone split for employment (p 0.19) and hours (p 0.86), with unemployment the exception (p 0.031), consistent with a link that is mostly the North-South difference. Track A alone cannot say this (it ranks Italy among countries), and Track B alone cannot say that Italy's local-unit slope is unusual; the two together single out firm structure for Italy and show governance as a within-country regularity that Italy shares. Track B BH q-values cover 172 pairs chosen by eye: 16 survive within zones (6 indicators: EQI, impartiality, local units, capital share, capital spending, local-government spending).
- **OpenCoesione 2014-2020: the funds follow the North-South split and nothing remains within zones.** South and Islands received 2,307 euro of EU funds per inhabitant against 485 in the North (4.8 times), 893 against 264 of national co-financing of the EU programmes (3.4 times) and 4,939 against 839 of the FSC (5.9 times), while the share paid out is 62% against 82%. Across the 21 regions all four predictors are strongly associated with every outcome (rho -0.68 to -0.77 with income, employment and hours, +0.78 to +0.80 with unemployment for the funds; the payment rate +0.62 to +0.74), and all 24 pairs are BH discoveries, but within zones none is (smallest within-zone p 0.063, rho -0.43 for the payment rate against unemployment; 0 of 24 at q <= 0.05) and the shuffled-predictor null gives a discovery in 4% of 100 shuffles in the unrestricted column (mean 0.53 discoveries) and 6% within zones (mean 0.29). So the data do not separate the allocation rule (funds go where income is low) from any effect of the funds; Sicily's low payment rate (38%) is visible but does not carry a within-zone pattern. Projects of several regions are split equally and 5% of the money has no Italian localisation. Descriptive.\n- **Limits.** EQI measures perceptions from surveys with margins of error, 2010-2024, Germany and Belgium are missing (NUTS1); a perception of institutions can follow prosperity as well as precede it. Structural business statistics end in 2020, persons employed include the self-employed (so pay per person is not pay per employee), and local units include one-person units. Spending is attributed to where it is incurred and includes central-government spending and pensions (28% of the total on average), so it is not what regional governments choose. Migration is registry transfers, with age only as a total (no education of movers). With 21 regions, 8 in the South and Islands and three zones, Track B has little power and its p-values are exploratory; the zone split removes real North-South differences as well as confounding. Nothing here separates cause from consequence: no claim that better institutions, firm structure, migration or spending cause the South's gap or close it.
"""))

NOTEBOOKS["35_regulation_productivity"] = [
    (M, """# Regulation and productivity: Italy in the OECD product market and employment protection indices
Candidate institutional explanations of Italy's productivity gap (notebooks 08, 13, 23): rules that limit competition (product market regulation, PMR) and rules on dismissals (employment protection, EPL). Data: the OECD indices via the SDMX API (`sources/oecd_regulation.py`; scale 0-6, higher = more restrictive).
- **PMR** exists here in two waves, **2018 and 2023**, for 57 economies (all EU27), with sector indicators for professions, retail, energy, transport and e-communications. There is no earlier wave in this flow, so it is a cross-section.
- **EPL** is annual, **1985-2019** (VERSION1; later years repeat), 21-22 EU members from the 1990s.
- **CompNet** (firm-level productivity dispersion) is only available on request from IWH Halle (form), so it is not used.
**Test (sector level, so there are more than 30 observations):** is the productivity growth of a sector in a country (log growth per year of real value added per hour, 2005-07 to 2017-19, EU KLEMS-style `sectors64`) related to the regulation of *that* sector in 2018? Six sectors have a matching PMR indicator (retail, legal and accounting, architecture and engineering, land transport, telecommunications, electricity and gas), 17-26 countries each. Within each sector both variables are ranks (so sector levels drop out); pooled correlation; p-value from shuffling the regulation across the countries of each sector; Benjamini-Hochberg over the pooled test and the six sector tests; shuffled-regulation null; Italy by placebo ranking, not by a test. PMR is measured at the *end* of the growth window, so the association cannot tell which came first. **Associations only, never causal.** Code: `src/cifresignificative/analysis/regulation.py`, `experiments.run_regulation_all`."""),
    (C, '''from cifresignificative.analysis import regulation as RGL, experiments as ex
res = ex.run_regulation_all()
pos = res["g1_pmr_position"]
print("PMR, Italy against the other EU27 members (rank 1 = most restrictive; waves 2018 and 2023):")
display(pos.pivot(index="measure", columns="year", values=["italy", "median_others", "rank"]).loc[RGL.POSITION].round(2))
e = res["g2_epl_gap"].set_index("year")
print("EPL overall (VERSION1), Italy vs the median of the other EU members present, and Italy's rank (1 = strictest):"); display(e.loc[[1985, 1990, 1995, 2000, 2005, 2010, 2015, 2019]].round(2).T)
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.plot(e.index, e.italy, color="#d94f3d", lw=2, label="Italy"); ax.plot(e.index, e.median_others, color="#2a78d6", lw=1.6, label="median of the other EU members present")
ax.set_ylabel("EPL overall, 0-6"); ax.set_title("Employment protection: Italy vs the others (VERSION1)"); ax.legend(fontsize=8); plt.tight_layout(); plt.show()'''),
    (M, "## G3 - does a sector's regulation go with its productivity growth?\nPrimary family: window 2005-07 to 2017-19 with PMR 2018 (pooled + six sectors, BH over these seven rows). The other rows are robustness (early and late halves of the growth window, PMR 2023) and are descriptive."),
    (C, '''g3 = res["g3_sector_test"]
print("primary family (rho = Spearman of growth on regulation within sector; negative = more regulated sectors grow more slowly):"); display(g3[g3.primary][["sector", "n", "rho", "p", "q_value", "significant"]].round(3))
print("pooled test by window and PMR wave (descriptive):"); display(g3[g3.sector == "pooled"][["window", "pmr_year", "n", "rho", "p"]].round(3))
print("sector tests with p < 0.05 in each specification:", g3[g3.sector != "pooled"].groupby(["window", "pmr_year"]).apply(lambda x: int((x.p < 0.05).sum())).to_dict())
from cifresignificative.sources import oecd_regulation as OR
d = RGL.sector_data(OR.load_pmr())
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for k, (code, grp) in enumerate(d.groupby("nace_r2")):
    ax.scatter(grp.rr, grp.rg, s=22, alpha=0.7, label=RGL.SECTORS[str(code)][0])
it = d[d.iso3 == "ITA"]; ax.scatter(it.rr, it.rg, s=90, facecolors="none", edgecolors="k", lw=1.5, label="Italy")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("PMR 2018 of the sector (within-sector standardised rank)"); ax.set_ylabel("productivity growth 2005-19 (within-sector rank)"); ax.legend(fontsize=7)
ax.set_title("Country x sector pairs (n = %d)" % len(d)); plt.tight_layout(); plt.show()
n0 = res["g6_null"].iloc[0]; print("shuffled regulation within sectors,", int(n0.n_rep), "repetitions: discoveries per repetition", round(float(n0.mean_discoveries), 3), "| repetitions with a discovery", round(float(n0.share_reps_with_discovery), 3), "| share of tests with p < 0.05", round(float(n0.mean_share_p_below_005), 3))'''),
    (M, "## G4 - where is Italy?\nItaly's regulation rank (1 = most restrictive) and productivity-growth rank (1 = fastest) in each sector, and a placebo ranking: for each country, the mean over its sectors of (growth rank minus the pooled slope times regulation rank); negative = slower growth than the pooled relation predicts."),
    (C, '''display(res["g4_italy_sectors"].round(2))
pl = res["g5_italy_placebo"]
print("placebo ranking of the mean gap (1 = most negative), Italy:", pl[pl.iso3 == "ITA"][["gap", "rank", "n_countries", "pooled_slope"]].round(3).to_dict("records"))
print("the five most negative:"); display(pl.head(5).round(3))'''),
]

NOTEBOOKS["35_regulation_productivity"].append((M, """## Takeaways (OECD PMR 2018 and 2023, EPL 1985-2019; sector productivity growth 2005-07 to 2017-19; descriptive associations)
- **Product market regulation: mid-table overall, restrictive in professions and retail, light in network industries.** Italy's overall PMR is 1.29 in 2018 (median of the other EU27 1.47, rank 16 of 27, 1 = most restrictive) and 1.23 in 2023 (median 1.22, rank 13). It is among the most restrictive in accountants (2.82 against a median of 0.08 to 0.22, rank 1 in both waves), architects (3.21, rank 2 then 1), services barriers (3.03 against 1.96, rank 4-5), retail trade (1.93 against 1.05, rank 6) and public procurement (rank 4, then 7 after a fall from 1.10 to 0.70), and among the least restrictive in network sectors (rank 25 of 27), energy (22), transport (23 to 26) and e-communications (21-22). Lawyers (rank 22-23) and notaries (19) are below the median. The professions' indicators are identical in the two waves (the rules did not change); the changes between waves are in network sectors, procurement and entry barriers.
- **Employment protection: high and constant until 2010, then lower.** The overall EPL index (VERSION1) is 3.02 from 1985 to 2010 (median of the other EU members present 2.68 falling to 2.36; rank 4-5), 2.93 in 2015 and 2.56 in 2019 (median 2.29, rank 7 of 22). The OECD's newer version (VERSION4, from 2013) has a one-year spike in 2021 for Italy (3.82 against 2.86 either side), driven by collective dismissals (6.00, the scale's maximum, against 3.19) and, less, regular contracts (2.94 against 2.72); temporary contracts do not move, no EU member shows it (the only other economy is Turkey, +3.05), and the 2020 and 2022 values are unchanged. That pattern fits the Covid-period ban on dismissals (2020-21; dates from memory) read at 1 January 2021, but the OECD's note was not checked and the series are not spliced.
- **No relation between a sector's regulation and its productivity growth.** Across 123 country x sector pairs (six sectors, 17-26 countries each) the pooled within-sector rank correlation between the 2018 PMR indicator and productivity growth is +0.06 (permutation p 0.49, q 0.74), and none of the six sectors is significant (smallest q 0.74; rho from -0.29 for legal and accounting to +0.38 for architecture and engineering, both n = 18-19). The sign is not stable: +0.23 in the early half of the window (p 0.012, in the wrong direction for a drag from regulation), -0.15 in the late half (p 0.095). Shuffling the regulation within sectors gives 0.07 discoveries per repetition (200 repetitions), so the procedure does not manufacture them. With about 123 pairs a correlation below roughly 0.18 in absolute value (1.96 / sqrt(122), an approximation) cannot be told from zero: the test rules out a strong sector-level link, not a weak one. PMR is measured at the end of the window, and the matched indicators are coarse (one rule set per profession, one retail index).
- **Italy's sectors grow slowly whatever their regulation.** Italy's productivity growth is 22nd of 26 in electricity and gas (-1.7% a year against a median of -0.1%), 17th of 19 in legal and accounting (-0.8% against +0.2%), 15th of 17 in land transport and 14th of 18 in architecture and engineering; it is mid-table in retail (13th of 18) and telecommunications (13th of 25). Its regulation rank in those sectors differs widely (2nd of 19 in legal and accounting, 4th in architecture and engineering, 3rd in retail, but 21st in electricity and gas), and the pooled slope is near zero. In the placebo ranking of the mean gap between growth rank and the pooled relation, Italy has the 3rd most negative gap of 18 countries (after Greece and Cyprus), which here only says that its sector productivity growth is low across these sectors.
- **Limits.** Two PMR waves only (no earlier vintage in this flow), measured at the end of the growth window; six sectors with matched indicators (marketed services and utilities, not manufacturing); within-sector sample sizes of 17-26; sector growth is computed from 2-digit value added per hour with missing hours for several countries; regulation and productivity can move together for other reasons (country size, development, measurement). CompNet (firm-level dispersion) is available only on request from IWH Halle. Nothing here is causal.
"""))

NOTEBOOKS["36_wage_setting"] = [
    (M, """# Wage-setting institutions and the pass-through of inflation to wages
Notebook 26 found that wage pass-through is weak and differs across countries, with none detectable in the 2021-23 energy shock. Here the cross-country differences are set against **collective-bargaining coverage** and **trade-union density** (OECD/AIAS, from the ICTWSS database, via the OECD SDMX API: `sources/oecd_bargaining.py`; per cent of employees, country means over 2005-2019).
**Design (one cross-section, so low power).** The country pass-through is the one of notebook 26 (monthly labour cost on inflation, 2005Q1-2021Q2, `hf.passthrough_by_country`, 27 countries). Two pre-specified institution variables; Spearman correlation with a **permutation p-value** (institution shuffled across countries), Benjamini-Hochberg over the two; leave-one-country-out; robustness without Greece and Portugal (negative estimates), with 2010-2019 means and with at least six observations per mean; shuffled null; Italy by position among the countries (rank), not by a test.
**Data notes.** Coverage is observed in few years for several countries (2-4 values for Estonia, Finland, Luxembourg, Latvia, Slovenia, Ireland, Slovakia), so its mean uses the years available; Italy's coverage is recorded as **100% in every year** (a top-coded point: minimum pay in collective contracts is extended by case law). Union density is observed in most years. **Associations only, never causal.** Code: `src/cifresignificative/analysis/bargaining.py`, `experiments.run_bargaining_all`."""),
    (C, '''from cifresignificative.analysis import bargaining as BGN, experiments as ex, hf as H
res = ex.run_bargaining_all()
inst = res["w1_institutions"].set_index("iso3")
pt = H.passthrough_by_country(H.load_highfreq(), ("2005-01-01", "2021-06-30")).set_index("iso3")
j = inst.join(pt.pass_through, how="inner").sort_values("pass_through")
print("countries with a pass-through estimate and an institution mean:", len(j), "| with coverage:", int(j.cbc.notna().sum()), "| with union density:", int(j.tud.notna().sum())); display(j.round(2).T)
print("primary tests and robustness (rho = Spearman with the country pass-through; negative = more bargaining goes with less pass-through):")
display(res["w2_association"][["spec", "variable", "n", "rho", "p", "q_value", "loo_min_abs_rho", "loo_sign_flips"]].round(3))
print("Italy among the countries with data:"); display(res["w3_italy"].round(2))
n0 = res["w4_null"].iloc[0]; print("shuffled pass-through,", int(n0.n_rep), "repetitions: discoveries per repetition", round(float(n0.mean_discoveries), 3), "| repetitions with a discovery", round(float(n0.share_reps_with_discovery), 3), "| share of tests with p < 0.05", round(float(n0.mean_share_p_below_005), 3))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, v, lab in [(axes[0], "cbc", "collective-bargaining coverage, % of employees (mean 2005-19)"), (axes[1], "tud", "trade-union density, % of employees (mean 2005-19)")]:
    for c, r in j.dropna(subset=[v]).iterrows():
        ax.scatter(r[v], r.pass_through, s=70 if c == "ITA" else 22, color="#d94f3d" if c == "ITA" else "#2a78d6"); ax.annotate(c, (r[v], r.pass_through), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel(lab); ax.set_ylabel("wage pass-through of inflation, 2005Q1-2021Q2")
fig.suptitle("Pass-through and wage-setting institutions (Italy in red; Greece at -1.96 is off the chart's lower edge)", fontsize=9); ax.set_ylim(-1.0, 1.2); axes[0].set_ylim(-1.0, 1.2)
plt.tight_layout(); plt.show()'''),
]

NOTEBOOKS["36_wage_setting"].append((M, """## Takeaways (OECD/AIAS bargaining coverage and union density, means 2005-2019; pass-through of notebook 26, 2005Q1-2021Q2; descriptive)
- **No relation between wage-setting institutions and pass-through across countries.** Across 21 countries with coverage and 23 with union density, the Spearman correlation with the country pass-through is 0.00 for coverage (permutation p 0.997) and -0.20 for union density (p 0.36, q 0.71 over the two tests). The robustness runs stay in the same range (coverage 0.03 to 0.17, union density -0.20 to -0.11; smallest p 0.36 over all eight rows). The sign of the coverage relation is not stable (removing a single country flips it in 14 of 21 leave-one-out runs). Shuffling the pass-through gives 0.04 discoveries per repetition (500 repetitions), so the procedure does not manufacture them.
- **Italy.** Coverage is recorded as 100% in every year, the joint highest with Belgium (median of the other countries 58%); union density is 32.1% (rank 6 of 23, median 18.0%). Its pass-through, 0.32, is below the median of the others (0.46; rank 15 of 23). The most extreme estimates are Greece (-1.96) and Portugal (-0.74); the tests use ranks, and dropping both changes nothing.
- **What the test can and cannot say.** With 21-23 countries only a rank correlation of about 0.4 or more (1.96 / sqrt(n - 1), an approximation) can be told from zero, and each country's pass-through is itself a noisy estimate (standard errors of 0.2-0.4 are common in notebook 26): a moderate link would not show. Coverage is observed in 2-4 years for several countries (means of few values), Italy's coverage is a top-coded point, and both institutions are slow-moving national averages that miss what matters for timing (contract renewal lags, sector agreements, indexation clauses).
- **Limits.** One cross-section, no sector or contract-level information, pass-through from monthly labour costs on HICP inflation (notebook 26), a third factor could drive both institutions and pass-through, and the energy-shock window (2021Q3-2023Q4) is too short for country estimates. Nothing here is causal.
"""))

NOTEBOOKS["37_nuts_spatial"] = [
    (M, """# NUTS 2021 geography and spatial dependence in the regional screens
Two limits of the regional work (notebooks 25 and 34) are checked here. **(1) Region codes mix NUTS classifications** (France's 2013 regions, Greece's 2010 codes, Istat's ITD/ITE, the 2024 codes of Portugal and the Netherlands ...): the codes are translated to NUTS 2021 where a region was only renamed, never invented where it was split. **(2) Neighbouring regions are not independent:** the tests demean by country and permute within countries, which absorbs country clustering but not the similarity of neighbours.

**Design.** *Crosswalk* (`clean/nuts.py`): area overlap of each old region with the 2021 regions on the GISCO 20M boundaries (2010, 2013, 2016, 2021, 2024); a region is `nested` when at least 95% of its covered area lies in one 2021 region (renamed or merged), `split` otherwise (left out), `later` when the code exists only from 2024 (left out); NUTS 2006 codes are mapped by hand (`LEGACY_2006`). Where Eurostat publishes both the old and the new code of a region in the same year the native row is kept and the difference is reported. *Spatial structure* (`analysis/spatial.py`): queen contiguity within a country (islands linked to the nearest region of the country), Moran's I of the country-demeaned columns, and **eigenvector spatial filtering** (Griffith): each column is purged of the smooth spatial patterns, chosen greedily, that carry its autocorrelation. The notebook 34 EU-wide family is rerun plain and filtered (same pairs, within-country permutation p, BH over the family). *Calibration:* every x is replaced by independent spatially smooth noise (autocorrelation 0.4 and 0.8) while y stays real; a calibrated pipeline rejects about 5% and makes about 0 discoveries.

**Reading the filter.** Filtering removes spatial gradients shared by both variables, including a North-South gradient inside a country, which is the thing "why the South" asks about. A pair that survives is associated beyond smooth spatial patterns; a pair that does not may owe its association to such a pattern (or lose it by over-filtering): neither answer is "the" answer. Descriptive associations only."""),
    (C, '''from cifresignificative.analysis import experiments as ex, regional_south as rs
from cifresignificative.analysis.data import PROCESSED
nres = ex.run_nuts_all()
cw3, cov, ov, n4 = nres["n3_crosswalk"], nres["n1_coverage"], nres["n2_overlap"], nres["n4_south_geography"]
reg, reg21 = pd.read_parquet(PROCESSED / "regional.parquet"), pd.read_parquet(PROCESSED / "regional_nuts2021.parquet")
print("region codes in regional.parquet:", reg.geo.nunique(), "| NUTS 2021 regions after harmonising:", reg21.geo.nunique(), "| renamed rows:", int(reg21.geo_method.eq("renamed").sum()), "of", len(reg21))
print("codes in use that are not NUTS 2021 codes, by kind:", cw3.kind.value_counts().to_dict())
print("renamed or merged (carried over):", ", ".join(f"{o}>{n}" for o, n in zip(cw3[cw3.kind.isin(["nested", "legacy"])].old, cw3[cw3.kind.isin(["nested", "legacy"])].new)))
print("split, not allocated:", ", ".join(cw3[cw3.kind.eq("split")].old), "| exist only from 2024:", ", ".join(cw3[cw3.kind.eq("later")].old))
print("both codes published in the same year, relative difference of the two values:"); display(ov.round(4))
print("regions with data per window (mean over the 26 indicators):"); display(cov.groupby("window")[["regions_before", "regions_after"]].mean().round(1))'''),
    (C, '''# the "why the South" family on the original and on the NUTS 2021 geography
g = n4.groupby(["wave", "income_control"]).agg(tests=("rho_original", "size"), survivors_original=("significant_original", "sum"), survivors_nuts2021=("significant_nuts2021", "sum"), same_sign=("rho_original", lambda s: float((np.sign(s) == np.sign(n4.loc[s.index, "rho_nuts2021"])).mean())), max_abs_rho_change=("rho_original", lambda s: float((s - n4.loc[s.index, "rho_nuts2021"]).abs().max())))
display(g.round(3))
prim = n4[n4.wave.eq(rs.PRIMARY_WAVE) & ~n4.income_control]
print("primary wave: n of regions per pair, original vs NUTS 2021:", int(prim.n_original.median()), "vs", int(prim.n_nuts2021.median()), "| pairs whose survival flips:", int((prim.significant_original != prim.significant_nuts2021).sum()), "of", len(prim))'''),
    (C, '''sres = ex.run_spatial_all()
m, fam, plc, nul = sres["sp1_moran"], sres["sp2_family"], sres["sp3_italy_rank"], sres["sp4_null"]
mp = m[m.wave.eq(rs.PRIMARY_WAVE) & ~m.income_control].set_index("indicator")
print("Moran's I of the country-demeaned columns (primary wave 2017, 2015-19 outcomes); z > 1.645 = positive spatial autocorrelation:")
display(mp[["n", "moran_i", "moran_z", "moran_i_filtered", "moran_z_filtered", "k_vectors"]].round(2))
print("columns with positive autocorrelation before / after filtering:", int((mp.moran_z > 1.645).sum()), "/", int((mp.moran_z_filtered > 1.645).sum()), "of", len(mp))'''),
    (C, '''s = fam.groupby(["wave", "income_control"]).agg(tests=("rho", "size"), survivors_plain=("significant", "sum"), survivors_filtered=("significant_filtered", "sum"), same_sign=("same_sign", "mean"))
display(s.round(2))
pf = fam[fam.primary & ~fam.income_control].copy(); pf["abs_rho"], pf["abs_rho_f"] = pf.rho.abs(), pf.rho_filtered.abs()
print("primary wave, plain survivors:", int(pf.significant.sum()), "| of them still significant after filtering:", int((pf.significant & pf.significant_filtered).sum()), "| filtered survivors that were not plain survivors:", int((~pf.significant & pf.significant_filtered).sum()))
print("median |rho|, plain vs filtered:", round(float(pf.abs_rho.median()), 3), "vs", round(float(pf.abs_rho_f.median()), 3), "| pairs with the sign changed:", int((~pf.same_sign).sum()))
print("survivors after filtering (primary wave, plain and beyond income):"); display(fam[fam.primary & fam.significant_filtered][["x", "y", "income_control", "n", "rho", "q_value", "rho_filtered", "q_value_filtered"]].round(3))'''),
    (C, '''print("calibration with independent spatially smooth x (real y): mean over simulations"); 
display(nul.groupby(["rho", "pipeline"]).agg(moran_i_of_x=("moran_i_x", "mean"), share_p_below_005=("share_p_below_005", "mean"), discoveries=("discoveries", "mean"), simulations_with_a_discovery=("discoveries", lambda d: float((d > 0).mean())), tests=("n_tests", "first")).round(3))
print("Italy's rank among countries (p_rank = share of countries with a slope at least as far from the others, 1/n_countries is the minimum), plain vs spatially filtered:")
it = plc.sort_values("p_rank"); display(it[["x", "y", "n_countries", "slope_diff", "p_rank", "slope_diff_filtered", "p_rank_filtered"]].head(12).round(3))
print("pairs with p_rank <= 0.10: plain", int((plc.p_rank <= 0.10).sum()), "| filtered", int((plc.p_rank_filtered <= 0.10).sum()), "of", len(plc))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
d = mp.sort_values("moran_i"); axes[0].barh(range(len(d)), d.moran_i, color="#d94f3d", alpha=.6, label="plain"); axes[0].barh(range(len(d)), d.moran_i_filtered, color="#2a78d6", alpha=.8, height=.4, label="filtered")
axes[0].set_yticks(range(len(d))); axes[0].set_yticklabels(d.index, fontsize=6); axes[0].axvline(0, color="k", lw=.5); axes[0].set_xlabel("Moran's I within countries"); axes[0].legend(fontsize=7)
axes[1].scatter(pf.rho, pf.rho_filtered, s=22, c=np.where(pf.significant, "#2a78d6", "#aaaaaa")); axes[1].plot([-1, 1], [-1, 1], "k", lw=.5); axes[1].set_xlabel("Spearman rho, plain"); axes[1].set_ylabel("Spearman rho, spatially filtered")
fig.suptitle("Spatial structure of the regional columns, and the 2017-wave family before and after filtering (blue = plain survivor)", fontsize=9); plt.tight_layout(); plt.show()'''),
]

NOTEBOOKS["37_nuts_spatial"].append((M, """## The notebook 25 screens on the NUTS 2021 geography, with and without spatial filtering
Notebook 25 screened every pair of regional indicators within countries (levels 2015-19, and changes between 2005-09 and 2015-19) with Benjamini-Hochberg control over all pairs; R2b replaced the asymptotic p by the within-country permutation p. The same two pair screens (`regional_south.screen_perm`, size-scaled columns and mechanical pairs excluded as before) are rerun here on (a) the original region codes, (b) the NUTS 2021 geography and (c) the NUTS 2021 geography after spatial filtering of every column. Calibration of the whole screen: columns shuffled within countries, and every column replaced by independent spatially smooth noise (autocorrelation 0.4 and 0.8, Moran's I of about 0.16 and 0.57), plain against filtered; the BH discoveries per screen should be about zero."""))
NOTEBOOKS["37_nuts_spatial"].append((C, '''scr = ex.run_spatial_screens(n_sim=30)
cnt, prs, nul = scr["sp6_screen_counts"].set_index("screen"), scr["sp5_screen_pairs"], scr["sp7_screen_null"]
print("pairs tested and BH survivors (within-country permutation p):"); display(cnt.T)
for name in ("levels", "changes"):
    x = prs[prs.screen.eq(name)]
    lost = x[x.significant & ~x.significant_filtered.fillna(False)]; gain = x[~x.significant.fillna(False) & x.significant_filtered.fillna(False)]; kept = x[x.significant & x.significant_filtered.fillna(False)]
    print(f"{name}: survivors on NUTS 2021 {int(x.significant.sum())}, kept after filtering {len(kept)}, lost {len(lost)}, new {len(gain)} | median |rho| of the lost {lost.rho.abs().median():.2f}, of all plain survivors {x[x.significant].rho.abs().median():.2f}, of the kept after filtering {kept.rho_filtered.abs().median():.2f} | kept pairs whose sign changed: {int((~kept.same_sign).sum())}")
    display(lost.sort_values("rho", key=abs, ascending=False)[["x", "y", "rho", "rho_filtered", "q_value", "q_value_filtered"]].head(6).round(3))
print("null of the whole screen (BH discoveries per simulation; share of tests with p < 0.05):")
display(nul.groupby(["screen", "null", "pipeline"]).agg(mean_discoveries=("discoveries", "mean"), simulations_with_a_discovery=("discoveries", lambda d: float((d > 0).mean())), share_p_below_005=("share_p_below_005", "mean"), tests=("n_tests", "first")).round(3))'''))

NOTEBOOKS["37_nuts_spatial"].append((M, """## Takeaways (NUTS 2021 geography and spatial dependence; notebook 34 family; descriptive)
- **Harmonising the codes changes almost nothing, and it is not a coverage fix.** 303 region codes become 242 NUTS 2021 regions: 56 old codes are carried over by a rename (46 by area overlap, 10 NUTS 2006 codes by hand: Istat's ITD/ITE and FI1A), 7 are splits left out (FI18, HR04, HU10, IE01, IE02, LT00, PL12) and 7 exist only from the 2024 classification (NL35, NL36, PT19, PT1A-D). 660 of 112,435 rows are renamed. France 2016 and Greece 2013 are pure relabelling, so no value is summed or allocated. Where Eurostat publishes both codes in the same year the values agree closely (physicians: median difference 0.24% over 436 pairs, 90th percentile 1.5%, maximum 14.5%; R&D: 2 pairs). The number of regions with data per window does not rise (2015-19: 236.6 codes before, 228.2 after, because the split and post-2021 codes drop out).
- **Notebook 34's EU-wide result does not depend on the code mix.** Survivors of the false-discovery cut are the same in every wave and variant except one extra survivor in the 2010 wave without income control (28 against 29); no pair flips in the primary wave (0 of 40), the sign agrees everywhere but one pair (2010, beyond income), and the largest change in rho is 0.09.
- **The regional columns are spatially autocorrelated, strongly for labour-market and governance variables.** Moran's I of the country-demeaned columns (2017 wave): long-term unemployment 0.65, youth unemployment 0.62, unemployment 0.61, employment 0.57, EQI 0.53, AROPE 0.52; the structural business statistics 0.14-0.16; GDP per head only 0.06 (z 1.2). 27 of 40 columns have a clearly positive I; after filtering none do (1-5 eigenvectors per column; 13 columns needed none). Two columns are negatively autocorrelated (physicians per 100,000, I -0.19; and fertility after filtering, -0.21): not addressed.
- **The within-country permutation test over-rejects under spatial autocorrelation; the filtered pipeline is close to calibrated.** With independent spatially smooth x and real y, the plain pipeline gives p < 0.05 in 10.7% of tests when x has Moran's I 0.16 (0.43 false discoveries per family, a discovery in 11% of simulations) and in 25.0% when I is 0.57 (4.08 discoveries per family, a discovery in 51% of simulations); the filtered pipeline gives 6.5% and 6.6% (0.06 and 0.11 discoveries, a discovery in 3-4% of simulations). Nominal level: 5% and about 0.
- **Filtering halves the primary-wave family without income control and leaves the beyond-income one intact.** 2017 wave, plain against filtered survivors: 37 against 15 without the income control (14 of the 37 stay, one pair is new: manufacturing share against female employment), 18 against 23 beyond income; the other waves 29/12, 32/25, 35/15 without and 20/22, 24/27, 20/20 beyond income (2010, 2013, 2021). Median |rho| falls from 0.316 to 0.156 and 5 pairs change sign. What stays without the income control in 2017: local units per 1,000 inhabitants against both employment rates, unemployment, GDP per head, output per hour and hours per inhabitant (rho 0.47-0.64, similar to or larger than the plain values), pay per employee against employment, unemployment and hours (smaller: 0.32 instead of 0.52 for employment), and EQI impartiality against unemployment and hours; **no other EQI pair survives, so the plain EQI-employment associations of notebook 34 are partly a spatial pattern**. Beyond income the EQI components survive against employment and unemployment (|rho| 0.18-0.35, mostly about half the plain values); a few beyond-income pairs change sign (EQI against GDP per head: 0.02 plain, -0.19 filtered), so treat them as fragile.
- **Italy's steep slopes do not survive filtering.** In notebook 34, 6 of 40 EU-wide pairs had Italy's slope among the most extreme of 14-16 countries (p_rank 0.062-0.067, the smallest possible value is 1/n). After filtering the columns, their ranks are 0.13-0.93 and only one pair of the whole family has p_rank <= 0.10 (pay per employee against unemployment, 0.071, which was 0.21 before). The Italian steepness is mostly the North-South gradient that both variables share, which is what the filter removes.
- **The notebook 25 screens survive spatial filtering.** On the original region codes the screens give the notebook 25 counts (164 of 284 pairs for the levels, 53 of 260 for the changes); on the NUTS 2021 geography 169 and 57; after spatial filtering of every column 159 and 46, with 152 and 44 pairs in both lists (17 and 13 lost, 7 and 2 new). The pairs lost are the weak ones (median |rho| 0.18 and 0.25 against 0.36 and 0.35 for all survivors) and involve broadband, old-age dependency and the labour share; no pair that stays changes sign (median |rho| of the kept pairs after filtering: 0.32 and 0.40). So the within-country associations of notebook 25 are not an artefact of the code mix or of neighbouring regions moving together. Spatial autocorrelation is nonetheless real in the columns (17 of 30 level columns and 10 of 29 change columns have a clearly positive Moran's I, none after filtering).
- **The whole-screen calibration depends on how smooth the noise is.** With columns shuffled within countries both screens give 0 discoveries. With independent spatially smooth noise of autocorrelation 0.4 (Moran's I about 0.16) the plain screen gives 0 discoveries (6.8-7.8% of tests at p < 0.05, filtered 5.9%); at 0.8 (Moran's I about 0.57, like the labour-market and governance columns) the plain screen makes 16.9 false discoveries per simulation for the levels (a discovery in 90% of 30 simulations) and 12.7 for the changes (70%), while the filtered screen makes none (7.6% and 7.2% of tests at p < 0.05, against 23-24% for the plain one). The real columns have a mix of the two regimes, which is why the plain survivors are mostly (not entirely) real.
- **How to read the filter.** It removes smooth spatial patterns, including a North-South gradient inside Italy, which is also what "why the South" asks about: a pair that survives is associated beyond such a pattern, a pair that does not may be explained by it or over-filtered. The 2017 family is therefore reported both ways and the plain test should be read as liberal for the autocorrelated columns.
- **Limits.** Boundaries are generalised (20M) and neighbours are same-country only (cross-border links ignored), the number of eigenvectors comes from a greedy rule with an analytic z-threshold (1.645), the filter is applied to each column separately, the area crosswalk is a proxy for population overlap and splits are not allocated, Germany and Belgium are absent from EQI, and the Italy-only track (21 regions) is not filtered because it has too few observations to spend degrees of freedom on eigenvectors. The calibration uses a first-order autoregressive pattern, not the real geography of the missingness. All associations are descriptive.
"""))

NOTEBOOKS["38_budget_laws"] = [
    (M, """# How the budget laws were financed, and what that says about the deficit and the debt
Each year the Italian government proposes a budget law (*manovra*) whose cost must be *covered*. The dossiers "Effetti sui saldi e conto risorse e impieghi" of the Camera and Senate budget services reproduce the government's technical report as a table: **impieghi** (uses: higher spending and lower revenue) against **risorse** (resources: higher revenue and lower spending); the balance is resources minus uses. This notebook measures the coverage of each manovra from 2016 to 2026 and then asks, as an exploratory exercise, whether the planned deficit effect lines up with what happened to the deficit and the debt.

**Data and checks.** 13 dossiers parsed by `sources/budget_laws.py`; every table satisfies the accounting identities (uses = higher spending + lower revenue, resources = higher revenue + lower spending, balance = resources - uses), and 2025 was also checked against the page image. 2018 is the initial bill; 2019 has totals only (no composition); **2022 has no published gross split** (the budget service published no "Effetti sui saldi" dossier for that law): the net figure comes from the summary prospetto of its Nota di lettura 274, and the uses and resources are **rebuilt from the article-by-article measure lines of the same prospetto** (each measure is a use if it raises spending or lowers revenue, a resource otherwise; the Sezione I lines reproduce the net totals and resources minus uses reproduce the net manovra, `evidence` = `reconstructed_from_measure_lines`), so its ratio is comparable with the others but not the government's own table (the result without it is the earlier one: see the takeaways). The 2020-21 pandemic decrees are separate legal acts and are kept apart.

**Measures** (net borrowing, first year of each law unless stated): coverage ratio = resources / uses (1 = fully covered inside the manovra, 0 = entirely financed by deficit); deficit-financed share = 1 - coverage; the deficit effect in % of GDP; the split of the resources into higher revenue and lower spending. Each manovra is attributed to the government in office on 15 October of the previous year (the draft budgetary plan), except the 2023 bill, which the Meloni government presented in November 2022 after taking office on 22 October.

**Link to the outcomes (exploratory).** A manovra can only act on the primary balance, and only by its planned amount, so debt/GDP is the weakest link by construction: nominal growth, interest and stock-flow items (the Superbonus tax credits, for instance) move it more. The pre-specified family has two predictors (planned deficit effect in % of GDP, deficit-financed share) against three outcomes of the same calendar year (change in net borrowing, change in the primary deficit, change in debt/GDP; all in percentage points of GDP, positive = looser), expected sign positive, Spearman with a permutation p, BH over the six tests. Checks: leave-one-year-out, without 2020-21, with the pandemic decrees added, a lead placebo (the previous year's outcomes) and a year-shuffle null that must give about zero discoveries. A second family asks whether the planned effect and the deficit-financed share go with the **deficit surprise** (outturn minus the programmatic deficit target the government gave for that year in its draft budgetary plan, `sources/deficit_targets.py`; two tests). With about ten years, only a rank correlation of about 0.65 or more can be told from zero. One country, no placebo countries: nothing here is a causal claim."""),
    (C, '''from cifresignificative.analysis import budget as BU, experiments as ex
bres = ex.run_budget_all()
first, three, T, fam = bres["b1_first_year"], bres["b2_three_year"], bres["b3_link_table"].set_index("budget_year"), bres["b4_family"]
fb = first[first.scope.eq("bill")].set_index("budget_year")
cols = ["government", "uses", "resources", "balance", "coverage_ratio", "deficit_financed_share", "uses_pct_gdp", "deficit_pct_gdp", "resources_from_revenue", "resources_from_spending", "uses_spending_share"]
print("first year of each law (EUR million; shares of the uses; GDP shares of the outturn nominal GDP):"); display(fb[cols].round(3))
print("same, summed over the three-year horizon (laws with all three years and a gross split):"); display(three[three.scope.eq("bill")].set_index("budget_year")[["uses", "resources", "balance", "coverage_ratio", "deficit_financed_share"]].round(3))
bud = first[first.scope.eq("budget")].set_index("budget_year"); print("bill plus its decree-law (2020: DL 124/2019, 2024: DL 145/2023), coverage ratio:", bud.coverage_ratio.round(3).dropna().to_dict())
pan = first[first.scope.eq("pandemic")].set_index("budget_year"); print("pandemic decrees:"); display(pan[["government", "uses", "resources", "balance", "coverage_ratio", "deficit_pct_gdp"]].round(3))
cr = fb.coverage_ratio.dropna()
print("coverage ratio, years 2016-2024 without 2021:", cr.drop(2021)[cr.drop(2021).index <= 2024].round(2).to_dict(), "| 2025:", round(cr[2025], 2), "| 2026:", round(cr[2026], 2))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
y = fb; x = np.arange(len(y))
axes[0].bar(x, y.resources_from_revenue.fillna(0), color="#2a78d6", label="higher revenue"); axes[0].bar(x, y.resources_from_spending.fillna(0), bottom=y.resources_from_revenue.fillna(0), color="#7fb2e8", label="lower spending")
axes[0].bar(x, y.deficit_financed_share, bottom=y.coverage_ratio, color="#d94f3d", label="financed by deficit")
axes[0].set_xticks(x); axes[0].set_xticklabels(y.index, rotation=0); axes[0].set_ylabel("share of the uses"); axes[0].set_title("How each law was covered (first year; 2019 has no split; 2022 rebuilt from the measure lines)", fontsize=9); axes[0].legend(fontsize=7)
a = T.loc[2016:2025]
axes[1].scatter(a.deficit_pct_gdp, a.delta_deficit, c=np.where(a.index.isin([2020, 2021]), "#d94f3d", "#2a78d6"))
for t, r in a.iterrows(): axes[1].annotate(str(t), (r.deficit_pct_gdp, r.delta_deficit), fontsize=7, xytext=(3, 2), textcoords="offset points")
axes[1].axhline(0, color="k", lw=.5); axes[1].set_xlabel("planned deficit effect of the manovra, % of GDP"); axes[1].set_ylabel("realised change in net borrowing, pp of GDP"); axes[1].set_title("Planned effect against the realised change in the deficit (red: 2020-21)", fontsize=9)
plt.tight_layout(); plt.show()'''),
    (C, '''dec = bres["b9_debt_decomposition"].set_index("year").loc[2016:2025]
print("realised outcomes by calendar year (pp of GDP; deficit and debt levels in % of GDP) next to the planned deficit effect:")
display(T.loc[2016:2025, ["deficit_pct_gdp", "deficit", "delta_deficit", "delta_primary_deficit", "delta_debt", "real_growth"]].round(2))
print("change in debt/GDP split into primary balance, interest, nominal growth and stock-flow (pp of GDP):"); display(dec[["delta_debt", "debt_primary", "debt_interest", "debt_nominal", "debt_other"]].round(1))
print("realised deficit over the planned deficit effect of the manovra:", (T.loc[2016:2025, "deficit"] / T.loc[2016:2025, "deficit_pct_gdp"]).round(1).to_dict())'''),
    (C, '''print("pre-specified family, 2016-2025 (rho = Spearman, p = permutation, q = BH over the six tests):"); display(fam[["x", "y", "n", "rho", "p", "q_value", "loo_min", "loo_max", "loo_sign_flips"]].round(3))
print("without 2020-21:"); display(bres["b5_family_ex_covid"][["x", "y", "n", "rho", "p", "q_value"]].round(3))
print("with the pandemic decrees added to 2020 and 2021:"); display(bres["b6_family_with_pandemic"][["x", "y", "n", "rho", "p", "q_value"]].round(3))
print("lead placebo (the manovra of year t against the outcomes of year t-1):"); display(bres["b7_lead_placebo"][["x", "y", "n", "rho", "p", "q_value"]].round(3))
nl = bres["b8_null"]; print("year-shuffle null,", len(nl), "shuffles: mean discoveries", round(float(nl.discoveries.mean()), 3), "| shuffles with a discovery", round(float((nl.discoveries > 0).mean()), 3), "| mean share of p < 0.05", round(float(nl.share_p_below_005.mean()), 3))
print("discoveries in the main family:", int(fam.significant.sum()), "of", len(fam), "| smallest q:", round(float(fam.q_value.min()), 3), "| signs of rho:", (np.sign(fam.rho)).astype(int).tolist())'''),
]

NOTEBOOKS["38_budget_laws"].append((C, '''tg = BU.targets()
print("programmatic deficit target of the budget year (draft budgetary plan or dossier), the outturn and the miss (pp of GDP, positive deficits; surprise = outturn - target, positive = a larger deficit than planned):")
display(T.loc[2016:2025, ["government", "deficit_pct_gdp", "target", "target_final", "deficit", "deficit_surprise", "deficit_surprise_final", "planned_delta_deficit", "delta_deficit"]].round(2))
print("second pre-specified family: the manovra against the deficit surprise (BH over the two tests):"); display(bres["b10_surprise_family"][["x", "y", "n", "rho", "p", "q_value", "loo_min", "loo_max"]].round(3))
print("without 2020-21:"); display(bres["b11_surprise_ex_covid"][["x", "y", "n", "rho", "p", "q_value"]].round(3))
print("against the revised target (2019: 2.04 instead of 2.4):"); display(bres["b12_surprise_final_target"][["x", "y", "n", "rho", "p", "q_value"]].round(3))
n2 = bres["b13_surprise_null"]; print("year-shuffle null of this family:", len(n2), "shuffles, mean discoveries", round(float(n2.discoveries.mean()), 3), "| shuffles with a discovery", round(float((n2.discoveries > 0).mean()), 3))
print("how well the target anticipated the change in the deficit (change implied by the plan = target - last year's outturn, against the realised change):"); display(bres["b14_plan_vs_outturn"].round(3))
fig, ax = plt.subplots(figsize=(7.5, 3.8)); a = T.loc[2016:2025]
ax.plot(a.index, a.target, "o--", color="#2a78d6", label="programmatic target"); ax.plot(a.index, a.deficit, "o-", color="#d94f3d", label="outturn (Eurostat, latest vintage)")
ax.set_ylabel("net borrowing, % of GDP"); ax.set_title("Deficit target of each budget law and what happened", fontsize=9); ax.legend(fontsize=7); plt.tight_layout(); plt.show()'''))

NOTEBOOKS["38_budget_laws"].append((C, '''dec = bres["b15_surprise_decomposition"].set_index("budget_year")
print("deficit surprise split into revenue shortfall, primary-expenditure overrun, interest overrun (EUR levels in % of the outturn GDP; revenue shortfall negative = revenue above plan) and a denominator effect; gdp_ratio = forecast / outturn nominal GDP:")
display(dec[["deficit_surprise", "revenue_shortfall", "primary_expenditure_overrun", "interest_overrun", "denominator_effect", "rounding", "gdp_ratio"]].round(2))
print("growth forecast errors (outturn minus the plan's forecast, pp):"); display(dec[["real_growth_forecast", "real_growth", "real_growth_error", "nominal_growth_forecast", "nominal_growth", "nominal_growth_error"]].round(1))
print("rank correlations with the nominal and real growth errors (permutation p):"); display(bres["b16_surprise_drivers"].round(3))
fig, ax = plt.subplots(figsize=(8, 3.9)); yrs = dec.index.to_numpy(); w = 0.2
for k, (c, lab, col) in enumerate([("revenue_shortfall", "revenue shortfall", "#d94f3d"), ("primary_expenditure_overrun", "primary expenditure overrun", "#2a78d6"), ("interest_overrun", "interest overrun", "#e0a030"), ("denominator_effect", "GDP denominator", "#999999")]):
    ax.bar(yrs + (k - 1.5) * w, dec[c], width=w, color=col, label=lab)
ax.plot(yrs, dec.deficit_surprise, "k_", ms=14, mew=2, label="deficit surprise"); ax.axhline(0, color="k", lw=.5); ax.set_ylabel("pp of GDP"); ax.set_title("What the deficit surprise was made of (positive = larger deficit than planned)", fontsize=9); ax.legend(fontsize=6, ncol=3); plt.tight_layout(); plt.show()'''))

NOTEBOOKS["38_budget_laws"].append((M, """### Which expenditure categories drove the overruns
The plan's target table splits total expenditure into eight rows (compensation of employees D.1, intermediate consumption P.2, social payments D.62/D.632, interest D.41, subsidies D.3, gross fixed capital formation P.51, capital transfers D.9 and a residual); Eurostat (`gov_10a_main`) has the outturn for the same categories. The miss of each category is computed as above (euro-level, in % of the outturn GDP, positive = more spending than planned) and the eight sum to the miss of total expenditure up to rounding. This attributes the overruns to *categories*, not to individual measures: a category mixes many measures, and the 2016 plan classified some items differently from later plans (intermediate consumption and social payments), so the 2016 split is not comparable."""))
NOTEBOOKS["38_budget_laws"].append((C, '''cat = bres["b17_expenditure_by_category"].set_index("budget_year")
print("expenditure miss by category (outturn minus target, euro-level, % of outturn GDP; positive = more spending than planned):"); display(cat.round(2))
big = cat.drop(columns=["total", "total_expenditure_miss"]).loc[2020:2023]
print("largest category each year:", {int(t): f"{r.idxmax()} ({r.max():.2f})" for t, r in big.iterrows()})
print("capital transfers in 2021-2023:", cat.loc[2021:2023, "capital transfers"].round(2).to_dict(), "| share of the total expenditure miss:", (cat.loc[2021:2023, "capital transfers"] / cat.loc[2021:2023, "total"]).round(2).to_dict())
print("max |sum of categories - miss of total expenditure|:", round(float((cat.total - cat.total_expenditure_miss).abs().max()), 3))
fig, ax = plt.subplots(figsize=(9, 3.9)); cc = cat.drop(columns=["total", "total_expenditure_miss", "interest"]).loc[2017:2025]; bottom_pos = np.zeros(len(cc)); bottom_neg = np.zeros(len(cc))
for c in cc.columns:
    v = cc[c].to_numpy(); ax.bar(cc.index, np.where(v > 0, v, 0), bottom=bottom_pos, label=c); ax.bar(cc.index, np.where(v < 0, v, 0), bottom=bottom_neg); bottom_pos += np.where(v > 0, v, 0); bottom_neg += np.where(v < 0, v, 0)
ax.axhline(0, color="k", lw=.5); ax.set_ylabel("pp of GDP"); ax.set_title("Expenditure misses by category, 2017-2025 (interest omitted; positive = more spending than planned)", fontsize=9); ax.legend(fontsize=6, ncol=3); plt.tight_layout(); plt.show()'''))

NOTEBOOKS["38_budget_laws"].append((M, """### The building credits inside the capital-transfer overruns: an ENEA lower bound and Istat's own figures
The capital-transfer overruns of 2021-23 are consistent with the Eurostat decision under which the Superbonus and building-bonus credits are capital expenditure when the works are done, but the category also holds other grants. ENEA's monthly bulletins give the Superbonus (110%) investments for completed works and the deductions (tax credits) accrued each December, cumulative (`data/external/enea_superbonus.csv`, transcribed from ENEA's annual reports with report and table; `analysis/superbonus.py`). Yearly flows are the differences between December snapshots (2021 is the cumulative since July 2020). Compared in euro with the capital-transfer overrun (pp of GDP times the outturn GDP). **This is a lower bound, not the amount:** ENEA covers the Superbonus only (no facade bonus, ordinary ecobonus, sismabonus or other building bonuses); the overrun is against the plan, which may already hold part of the credits; snapshots are by transmission date, so late transmissions move works between years (the flow of one year is understated, the next overstated). **Istat's own figures.** For the whole Superbonus (super ecobonus and super sismabonus) Istat gave the Senate budget committee on 26 April 2024 the effect on net borrowing as 0.03, 0.9, 2.8 and 3.9% of GDP in 2020-23, over 150 billion euro in total and over 175 billion with the facade bonus (about 23 billion for 2020-22 by difference); the numbers were first taken from the Camera's Dossier OFP n. 4 and then **checked against the note itself** (page 6 and footnote 7; the PDF is kept in `data/external/source/istat_note/`; `data/external/istat_superbonus_deficit.csv`). The note adds that only 3.6% of these credits could be used in deduction (the rest is invoice discount or cession) and, quoting the MEF, that the credits of the first cessioni reached 160.5 billion euro by 2023 (before cancellations and seizures). In euro they are the percentage times the outturn GDP, so they can differ by rounding from the dossier's own euro figures. Two further **primary** checks were added: Istat's hearing of March 2023 at the Senate finance committee (the vintage in which the credits were first booked in full, with a chart of the spending for the Superbonus and facade bonus together, read from the bars, about +-0.5 billion) and the Bank of Italy's memorandum of 22 April 2024 (credits matured of about 1%, almost 3% and 3.7% of GDP in 2021-23, almost 8% together)."""))
NOTEBOOKS["38_budget_laws"].append((C, '''fl = bres["b18_superbonus_flows"].set_index("budget_year"); lb = bres["b19_superbonus_lower_bound"].set_index("budget_year")
print("ENEA Superbonus flows by year (EUR billion): investments admitted, investments for completed works, deductions (credits) accrued, credit rate:"); display((fl[["investments_admitted_eur", "investments_completed_eur", "deductions_accrued_eur"]] / 1e9).join(fl.credit_rate).round(2))
print("against the capital-transfer overrun (EUR million; shares of the overrun):"); display(lb[["capital_transfers_miss_pct_gdp", "capital_transfers_miss_meur", "superbonus_credits_meur", "credits_pct_gdp", "share_of_capital_transfers_miss", "share_investment_only", "share_of_expenditure_miss", "remainder_meur"]].round(3))
ib = bres["b20_superbonus_istat"].set_index("budget_year")
print("Istat's own Superbonus effect on net borrowing (April 2024 note, as quoted by the Camera dossier), in euro, against the same overrun and against the ENEA credits:"); display(ib.round(3))
vt = bres["b21_superbonus_vintages"].set_index("budget_year")
print("alternative sources and vintages side by side (EUR million; the columns are not parts of one sum): ENEA credits, Istat April 2024 (Superbonus), Istat March 2023 (Superbonus + facade, read from a chart, 2021-22), Bank of Italy credits matured (% of GDP x GDP), Istat March 2023 spending on all firms' tax credits (context):"); display(vt.round(3))
ca = bres["b22_credit_attribution"].set_index("budget_year")
print("2021 and 2022 (Istat March 2023 vintage): overrun minus the building bonuses, then minus the firms' investment-type tax credits (core = Transizione 4.0 + Investimenti Mezzogiorno; wide adds R&D, DTA, vehicle bonus; Tavola 1 shares of the firm credit total), and the subsidy-type credits against the subsidies overrun (EUR million):"); display(ca.round(3))
sp = bres["b23_capital_transfer_split"].set_index("year")
print("Italy's capital transfers split into investment grants (D.92: Superbonus, facade bonus, firms' investment credits, PNRR investment grants) and other capital transfers (D.99: cash grants such as the pandemic contributi a fondo perduto), % of GDP and EUR million, with the change since 2019:"); display(sp[["d92_pct_gdp", "d99_pct_gdp", "d92_meur", "d99_meur", "d92_change_since_2019_meur", "d99_change_since_2019_meur", "building_plus_core_credits_meur", "building_plus_wide_credits_meur", "unexplained_core_meur", "unexplained_wide_meur"]].round(1))
gc = bres["b24_grants_check"].set_index("item").value
print("the 2021 cash grants of the Sostegni decrees (authorized in the technical reports; data/external/covid_grants_2021.csv) against the other capital transfers (D.99) and the unexplained part:"); display(gc.round(3).to_frame("value"))
print("grants at the payment rate of the automatic DL 73 tranche (paid over authorized in the law as converted):", round(float(gc.automatic_paid_over_authorized), 3), "->", round(float(gc.grants_at_payment_rate_meur)), "million; D.99 other than base and grants:", round(float(gc.d99_other_meur)), "million authorized, ", round(float(gc.d99_other_at_payment_rate_meur)), "million at the payment rate")
print("share of the overrun inside the capital-transfer category, 2021-23 together:", round(float(lb.loc["2021-23", "share_of_capital_transfers_miss"]), 3), "| credits", round(float(lb.loc["2021-23", "superbonus_credits_meur"]) / 1000, 1), "bn against an overrun of", round(float(lb.loc["2021-23", "capital_transfers_miss_meur"]) / 1000, 1), "bn")'''))
NOTEBOOKS["38_budget_laws"].append((M, """### Istat's own accounts by receiving sector (esploradati)
The previous sections used a judgment-based grouping of firms' credits and Eurostat's total for capital transfers. Istat's annual account of the general government (esploradati SDMX, dataflow `95_815_DF_DCCN_FPA_1` and `5_815_DF_DCCN_FPA_B19_1`; `sources/istat_fpa.py`, committed table `data/external/istat_fpa_s13.csv`, every edition from 2019 to October 2026) gives investment grants (D.92) and other capital transfers (D.99) **by the sector that receives them** (households, corporations, non-profit institutions, rest of the world), and keeps each publication (edition) so that the March 2023 reclassification can be seen directly. Receivers are Istat's classification; flows are accrual."""))
NOTEBOOKS["38_budget_laws"].append((C, '''rc = bres["b25_fpa_receivers"].set_index("year"); ri = bres["b26_fpa_increase"].set_index("year")
print("Istat, edition 2026M10: capital transfers paid by the general government by receiving sector (EUR million):"); display(rc.loc[2015:2025, ["d92_households", "d92_corporations", "d99_corporations", "d99_households", "d92_total", "d99_total", "subsidies"]].round(0))
print("rise over 2019 (EUR million) and, for 2021, the share of the total rise of D.92 + D.99:"); display(ri.loc[2020:2024, ["d92_households", "d92_corporations", "d99_corporations", "d99_households", "d92_total", "d99_total"]].round(0))
tot21 = float(ri.loc[2021, "d92_total"] + ri.loc[2021, "d99_total"]); print("2021 shares of the rise:", {k: round(float(ri.loc[2021, k]) / tot21, 3) for k in ("d92_households", "d92_corporations", "d99_corporations")}, "| rise", round(tot21))
rcl = bres["b27_fpa_reclassification"]; print("the March 2023 reclassification: the same year in the edition of October 2022 and in that of March 2023:"); display(rcl.round(0))
print("Istat's hearing of March 2023 (chart, Superbonus + facade) against the households' investment grants of the March 2023 edition above their 2019 level:"); display(bres["b28_fpa_hearing_check"].round(0))
fs = bres["b29_fpa_facade_split"]; print("Italy's deficit in successive Istat editions (% of the latest nominal GDP, so that only the vintage of the numerator changes) and its surprise against the programmatic target:"); display(bres["b31_fpa_deficit_vintages"].round(2))
sv = bres["b32_surprise_vintages"].set_index("budget_year")
print("deficit surprise and its parts with Istat's FIRST release of each year (the first edition that contains it) and with the latest edition (pp of the latest GDP; revision = latest minus first):"); display(sv[["first_edition", "deficit_surprise_first", "deficit_surprise_latest", "deficit_surprise_revision", "revenue_shortfall_first", "revenue_shortfall_latest", "primary_expenditure_overrun_first", "primary_expenditure_overrun_latest", "primary_expenditure_overrun_revision"]].round(2))
qp = bres["b33_quarterly_capital_profile"]; qr = bres["b34_quarterly_reclassification"]
print("quarterly account (raw series, edition 2026M10): other capital expenditure (D.92 + D.99), net lending and gross fixed capital formation by quarter, EUR million; share of the year:"); display(qp[["period", "other_capital_expenditure", "ooke_share_of_year", "net_lending", "gross_fixed_capital_formation"]].round(2))
print("how the March 2023 booking was spread over the quarters (quarterly edition of April 2023 minus that of January 2023), EUR million:"); display(qr.pivot(index="period", columns="item", values="change").round(0))
rz = bres["b36_superbonus_zones"]; rt = bres["b37_superbonus_rank_tests"]; rr = bres["b35_superbonus_regions"]
print("cross-check: Istat's latest edition against the Eurostat figures used for Italy (gap = Istat minus Eurostat, points of GDP, 2005-2024):"); display(bres["b38_istat_eurostat_check"].round(3))
print("who received the Superbonus: ENEA investments for completed works by zone (shares against population shares), snapshots end 2023 and end 2025:"); display(rz.round(3))
print("investment per inhabitant against GDP per inhabitant (Spearman over 20 regions, permutation p) and the condominium share:"); display(rt.round(3))
print("end-2023 snapshot by region, per inhabitant (EUR):"); display(rr[rr.snapshot.eq("2023-12-31")].sort_values("per_inhabitant_eur", ascending=False)[["region", "zone", "investment_meur", "per_inhabitant_eur", "gdp_per_inhabitant_eur", "condominium_share"]].round(2))
print("implied annual split of the households' investment grants above 2019 into the Superbonus (Istat note of 26 April 2024) and the facade bonus (rest); sum of the implied facade bonus 2020-22:", round(float(fs[fs.year <= 2022].implied_facade_meur.sum())), "million"); display(fs.round(0))'''))
NOTEBOOKS["38_budget_laws"].append((M, """### Who used the Superbonus, and when: the ENEA monthly reports by region
The annual reports give only two regional snapshots. ENEA also publishes a monthly report (62 month ends, August 2021 to September 2026; 22 pages each: Italy, the 20 regions, a summary), and `sources/enea_monthly.py` parses all of them into `data/external/enea_superbonus_monthly.csv` (admitted investment, investment of the completed works, deductions accrued, and the same by type of building; cumulative since July 2020, by transmission date, SuperEcobonus only). Checks: in every month the 20 regions add up to Italy (largest gap 56 thousand euro, March 2023) and the national totals match the transcribed snapshots of the annual reports at the same bulletin basis.
The flow of a window is the difference of two month ends. **Pre-specified family**: for each of 8 windows, the Spearman correlation across the 20 regions between the admitted investment per inhabitant of the window and GDP per inhabitant, permutation p, Benjamini-Hochberg over the windows, and a placebo that shuffles the regional incomes; the completed-works version and the zone shares are robustness. Descriptive: the income gradient says who used the credit, not why.
""")); NOTEBOOKS["38_budget_laws"].append((C, '''zw = bres["b39_superbonus_zone_windows"]; gw = bres["b40_superbonus_window_gradient"]; pb = bres["b41_superbonus_gradient_placebo"]; gp = bres["b42_superbonus_gradient_path"].set_index("date").rho
mf = bres["b43_superbonus_monthly_flows"].set_index("date"); cm = bres["b44_superbonus_completion"].set_index("date"); nb = bres["b45_superbonus_basis_national"]; rb = bres["b46_superbonus_basis_regions"]; ck = bres["b47_superbonus_monthly_checks"].set_index("date")
print("checks: largest |regions - Italy| over the 62 months (EUR):", ck.filter(like="gap_").abs().max().round(0).to_dict(), "| months with a regional decrease of the cumulative completed investment:", int((ck.n_decreases > 0).sum()), "(ENEA revisions, the largest in May 2026)")
print("monthly report against the national snapshots transcribed from the annual reports (EUR million, completed works):"); display(nb.round(3))
print("annual regional table (works ended by 31 December, later extraction) against the monthly report of the same month end, by region:"); display(rb.round(3))
print("share of the window's new admitted investment by zone against the population share (ratio 1 = in proportion):"); display(zw.pivot(index="window", columns="zone", values="ratio_of_shares").reindex(zw.window.unique()).round(2))
print("flow of admitted investment by window, EUR million:", zw.groupby("window", sort=False).flow_meur.sum().round(0).to_dict())
print("pre-specified family: income gradient of the new works, 8 windows (Spearman over 20 regions, permutation p, BH q):"); display(gw[gw.basis.eq("admitted")].drop(columns="basis").round(3).reset_index(drop=True))
print("robustness, completed works:"); display(gw[gw.basis.eq("completed")].drop(columns="basis").round(3).reset_index(drop=True))
print("placebo (regional incomes shuffled, whole family rerun): share of shuffles with any discovery at q < 0.05, mean discoveries:", pb.round(3).to_dict("records")[0])
print("cumulative gradient (rho of investment per inhabitant against income) at selected month ends:", {str(k.date()): round(float(v), 2) for k, v in gp.loc[["2021-08-31", "2022-06-30", "2022-11-30", "2022-12-31", "2023-01-31", "2023-03-31", "2023-12-31", "2024-12-31", "2026-09-30"]].items()})
print("share of the admitted investment already completed, by zone:"); display(cm.round(3))
print("largest months of new admitted investment (EUR million):", mf.sum(axis=1).sort_values(ascending=False).head(4).round(0).to_dict(), "| by calendar year:", mf.sum(axis=1).groupby(mf.index.year).sum().round(0).to_dict())
zcol = {"North": "#1f77b4", "Centre": "#7f7f7f", "South and Islands": "#d62728"}
fig, ax = plt.subplots(1, 2, figsize=(12, 3.9)); mf.plot.bar(stacked=True, ax=ax[0], width=0.9, color=[zcol[c] for c in mf.columns]); ax[0].set_xticks(range(0, len(mf), 6)); ax[0].set_xticklabels([str(d.date())[:7] for d in mf.index[::6]], rotation=45, ha="right")
ax[0].set_title("New admitted investment per month (EUR million), by zone"); ax[0].set_xlabel("")
ax[1].plot(gp.index, gp.values, color="#1f77b4"); ax[1].axhline(0, color="k", lw=0.5); ax[1].set_title("Income gradient of the cumulative investment per inhabitant (Spearman, 20 regions)"); ax[1].set_ylabel("rho")
plt.tight_layout(); plt.show()
'''))
NOTEBOOKS["38_budget_laws"].append((M, """### Did the regions that used the Superbonus more change differently? The link to the regional outcomes of notebook 34
ENEA's investment for completed works per inhabitant (end-2023 snapshot, 20 regions; Trentino-Alto Adige sums its two NUTS2 regions) against the change of the notebook 34 outcomes between 2019 and 2023 (Eurostat NUTS2; rates in percentage points, income and output per hour and hours per inhabitant in log points x 100). **Pre-specified**: Spearman of the investment per inhabitant against the change of the employment rate 20-64, permutation p over regions. **Placebos**: the same outcome over 2015-2019, which the credit cannot have affected; the investment shuffled within the three zones; and the whole 12-test family (6 outcomes x 2 periods, BH) rerun on shuffled investments. Robustness (not a family): investment per euro of 2019 GDP, and the rank correlation beyond the 2021 income level and the 2015-19 change. Descriptive: the credit followed the housing stock, ownership and the ability to advance the cost.
""")); NOTEBOOKS["38_budget_laws"].append((C, '''ol = bres["b49_superbonus_outcome_link"]; orb = bres["b50_superbonus_outcome_robustness"]; opl = bres["b51_superbonus_outcome_placebo"].round(3).to_dict("records")[0]
print("pre-specified test and the 12-test family (rho over 20 regions, permutation p over regions and within zones, BH q); 'before' = 2015-19 placebo window:"); display(ol.round(3))
print("robustness for the primary outcome:"); display(orb.round(3))
print("placebo (investment shuffled across regions, family rerun):", opl)
print("regions with the most and the least investment per inhabitant: employment-rate change 2019-23 and 2015-19 (pp):"); oc = bres["b48_superbonus_outcome_changes"].set_index("region"); rr2 = rr[rr.snapshot.eq("2023-12-31")].set_index("region").per_inhabitant_eur
display(pd.DataFrame({"investment_eur_per_inhabitant": rr2, "employment_change_2019_23": oc["employment_rate_20_64|after"], "employment_change_2015_19": oc["employment_rate_20_64|before"]}).sort_values("investment_eur_per_inhabitant", ascending=False).iloc[[0, 1, 2, 3, -4, -3, -2, -1]].round(2))
'''))
NOTEBOOKS["38_budget_laws"].append((M, """### Dating the policy changes of the winter 2022-23 against the monthly reports
`data/external/superbonus_policy_dates.csv` lists the dated rule changes (each date from two independent institutions, or labelled `partly verified`): the Aiuti-quater decree of 18 November 2022 (DL 176/2022, GU n. 270) with its 25 November cut-off for the CILA, the budget law of 29 December 2022, the stop to credit transfer and invoice discount of 16-17 February 2023 (DL 11/2023), the MEF note of 30 March 2023 on the single-family extension and the conversion law of 11 April 2023. ENEA counts the investment when the works are transmitted, so a rule on the start of works can show up only with a lag: the timing can rule an explanation out, not date its effect. **Pre-specified**: two events (the first full month after the cut-off, December 2022, and after the end of transfers, March 2023) x two measures (the income gradient of the month's admitted investment, and the national monthly flow): contrast of the six months from the cut against the five before, ranked among every possible cut from January 2022 to March 2026 (a rough yardstick: months are serially dependent and other rules sit among them), BH over the four. The gradient by building type shows what it is made of.
""")); NOTEBOOKS["38_budget_laws"].append((C, '''pdts = bres["b52_superbonus_policy_dates"]; tp = bres["b53_superbonus_monthly_path"].set_index("date"); te = bres["b54_superbonus_event_tests"]; on = bres["b55_superbonus_onset"].round(0).to_dict("records")[0]
print("dated rule changes (evidence label; every verified date has at least two sources):"); display(pdts[["date", "date_type", "title", "instrument", "evidence"]])
print("pre-specified event tests (D = mean after six months minus mean before five; rank among every cut, BH q):"); display(te.round(3))
print("onset:", on)
nbld = bres["b56_superbonus_new_buildings"].set_index("date"); nbt = bres["b57_superbonus_new_buildings_tests"]
print("exploratory proxy for starts (no project-level CILA dates are published): new buildings entering the ENEA report per month, by type:"); display(nbld.loc["2022-07-31":"2023-06-30"].round(0).T)
print("log-ratio contrasts of the new-building counts (six months from the cut against five before), rank among every cut month; not part of the pre-specified family:"); display(nbt.round(3))
print("single-family monthly flow after the extension was approved (27 Mar 2023, deadline 31 Mar -> 30 Sep), EUR million:", tp["single-family|flow_meur"].loc["2023-02-28":"2023-10-31"].round(0).to_dict())
g = tp[[c for c in tp.columns if c.endswith("|rho")]].loc["2022-01-31":"2023-12-31"]; g.columns = [c.split("|")[0] for c in g.columns]
print("income gradient of the month's admitted investment per inhabitant (Spearman over 20 regions) by building type:"); display(g.loc["2022-07-31":"2023-06-30"].round(2).T)
fl = tp[[c for c in tp.columns if c.endswith("|flow_meur")]].loc["2022-01-31":"2023-12-31"]; fl.columns = [c.split("|")[0] for c in fl.columns]
print("monthly admitted investment by building type, EUR million:"); display(fl.loc["2022-07-31":"2023-06-30"].round(0).T)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.9)); fl[["condominiums", "single-family", "independent units"]].plot.bar(stacked=True, ax=ax[0], width=0.9, color=["#1f77b4", "#d62728", "#7f7f7f"]); ax[0].set_xticks(range(0, len(fl), 3)); ax[0].set_xticklabels([str(d.date())[:7] for d in fl.index[::3]], rotation=60, fontsize=7); ax[0].set_title("New admitted investment per month (EUR million)"); ax[0].set_xlabel("")
for k in ["all", "condominiums", "single-family"]: ax[1].plot(g.index, g[k], label=k)
for d, lab in [("2022-11-25", "25 Nov"), ("2023-02-17", "17 Feb")]: ax[1].axvline(pd.Timestamp(d), color="k", ls=":", lw=0.8); ax[1].text(pd.Timestamp(d), -0.7, lab, fontsize=7, rotation=90, va="bottom")
ax[1].axhline(0, color="k", lw=0.5); ax[1].legend(fontsize=7); ax[1].set_title("Income gradient of each month's new investment (rho)")
plt.tight_layout(); plt.show()
'''))
NOTEBOOKS["38_budget_laws"].append((M, """### Re-basing the lower bound on the monthly national series
The lower bound above uses the December snapshots transcribed from the annual reports. The monthly reports (Italy row at each December month end) are later extractions: the same cumulative, with late transmissions in. Re-basing moves credits between years and leaves the three-year total where it was when the last snapshot is the same.
""")); NOTEBOOKS["38_budget_laws"].append((C, '''print("flows from the monthly reports (Italy row, December month ends), EUR:"); display(bres["b58_superbonus_rebased_flows"].round(3))
print("lower bound of the capital-transfer overrun on the two bases:"); display(bres["b59_superbonus_rebased_bound"].round(3))
'''))
NOTEBOOKS["38_budget_laws"].append((M, """## Takeaways (resources-and-uses tables of the budget-law dossiers, 2016-2026; net borrowing, first year unless stated; exploratory and descriptive)
- **Between 2016 and 2024 about half of each manovra was financed by deficit.** The coverage ratio (resources over uses) of the bill is 0.49-0.65 in 2016-2020 and 2023-2024 (2016 0.49, 2017 0.65, 2018 0.59, 2019 0.50, 2020 0.49, 2023 0.50, 2024 0.52), so 35% to 51% of the uses were not covered inside the law; 2021 (0.20, the pandemic budget) and 2022 (0.37, rebuilt from the measure lines) are the exceptions. 2025 (0.74) and 2026 (0.96) are the most covered. Summed over the three-year horizon the ratio runs from 0.30 (2022) to 0.80 (2023), 0.61 for 2025 and 0.79 for 2026: the coverage of the two latest laws is concentrated in the first year (0.74 against 0.61; 0.96 against 0.79), while for 2017, 2023 and 2024 the three-year ratio is the higher one.
- **Size.** The planned deficit effect of the bill is between 0.39% of GDP (2025) and 1.34% (2021); outside 2021 the largest is 2019 (1.21%, Conte I). The 2019 figures are those of the initial text: the dossier's table for the final texts has a balance of -11.5 billion on net borrowing in 2019 against -21.8 billion initially, so these series describe what each government *proposed*, and the approved law can be much smaller. Adding the decree-law that goes with the bill (2020, 2024) moves the ratio from 0.49 to 0.57 and from 0.52 to 0.57.
- **What covers the manovra.** Lower spending is the larger source in 2016 (0.29 against 0.20 from higher revenue), 2017 (0.35 against 0.30) and above all 2025 (0.50 against 0.24); higher revenue dominates in 2018 (0.40 against 0.18) and 2026 (0.43 against 0.30). "Lower spending" is a planned reduction against the baseline, not a cut from the previous year's spending. 2019 has totals only (a ratio, no composition). 2022 has no published gross split: its uses (36.8 billion) and resources (13.5 billion) are rebuilt from the measure lines of the Senate budget service's note, giving a ratio of 0.37, higher revenue 0.22 of the uses and lower spending 0.15 (deficit effect -23.3 billion, 1.16% of GDP); this sums every measure line as a use or a resource, so it is comparable with the dossiers' tables but is not the government's own table.
- **The pandemic decrees are a different object.** The 2020 decrees have uses of 115.4 billion and resources of 10.2 billion (coverage 0.09, a deficit effect of 6.3% of GDP); DL 41/2021 covers 0.03 (1.7% of GDP).
- **No link between the planned deficit effect and the realised deficit or debt can be shown.** 0 of 6 tests of the pre-specified family survive false-discovery control (smallest q 0.182) and all six correlations are negative (-0.14 to -0.64), the opposite of the expected sign. The strongest are against the change in debt/GDP: the planned deficit effect, rho -0.62 (p 0.061, q 0.182), and the deficit-financed share, -0.64 (p 0.052, q 0.182); without 2020-21 -0.67 (q 0.25) and -0.81 (p 0.022, q 0.13); with the pandemic decrees added to 2020 and 2021 the six correlations lie between -0.47 and -0.02. The year-shuffle null gives 0.087 discoveries per shuffle and a discovery in 6% of 300 shuffles (nominal about 5%), and the lead placebo (the manovra against the previous year's outcomes) finds nothing either (rho 0.01 to 0.35, smallest q 0.78). Before the 2022 gross split was rebuilt, the 2022 law was missing from the three tests on the deficit-financed share (n 9) and the smallest q was 0.365: the conclusion, 0 of 6, is the same.
- **Why this is not a surprise.** The planned effect of a bill (0.4-1.3% of GDP) is small next to what moves the deficit and the debt: nominal growth lowered debt/GDP by 14.5, 11.6 and 9.3 points in 2021-2023 (the ratio fell by 8.6, 7.4 and 4.5 points in total) while the planned effects were the largest of the period (1.34%, 1.16%, 0.99%), and the ratio rose in 2024-2025 (+0.8, +2.4 points) when they were small (0.71%, 0.39%); interest adds 3.1 to 3.8 points every year and the stock-flow residual +1.7 and +2.6 points in 2024 and 2025. The negative signs reflect the timing of the growth shock and not an effect of the manovra: with about ten years the correlation does not separate them.
- **The deficit targets were missed in both directions.** The outturn was above the target in 2016-2018 (by 0.2, 0.2 and 0.6 points of GDP), below it in 2019 (-0.9 against the initial 2.4, -0.54 against the revised 2.04), far above in 2020 (+7.2), and above again in 2021-2023 (+1.9, +2.5, +2.6) before coming in below in 2024 (-0.9) and 2025 (-0.2). The outturns are the latest Eurostat vintage, so they include later revisions that the targets could not know about.
- **The size and the financing of the manovra do not line up with the miss.** The planned deficit effect against the deficit surprise: rho 0.29 (p 0.41, q 0.41) and the deficit-financed share 0.49 (p 0.16, q 0.32), both positive as expected but not significant; without 2020-21 they are 0.07 and 0.35 (q 0.87 and 0.79); against the revised target 0.32 and 0.49 (q 0.37 and 0.32). The year-shuffle null gives a discovery in 5% of shuffles. The change that the target implied (target minus last year's outturn) went with the realised change in the deficit: rho 0.75 (p 0.014, same sign in 8 of 10 years), 0.62 without 2020-21 (p 0.107, same sign in 6 of 8): the targets carried information about direction, which says nothing about the manovra itself.
- **What the misses were made of.** Measured as euro levels in % of the outturn GDP, with the effect of the GDP forecast error kept apart, the 2020 miss (+7.2) is 3.4 points of revenue shortfall and 3.8 of primary-expenditure overrun (interest -0.2, GDP denominator +0.2). In 2021, 2022 and 2023 (+1.9, +2.5, +2.6) primary expenditure ran 3.3, 3.7 and 3.8 points above the plan, partly offset by revenue above the plan (shortfall -1.6, -2.2, -0.7), with interest 1.35 points above plan in 2022. The two years below target had revenue above plan (-2.2 in 2024, -1.1 in 2025) and spending above plan too (+1.6, +0.9). The denominator effect is small (at most 0.31 points in absolute value) and moves with the nominal GDP error as it must (rho -0.90, p 0.001). Nominal GDP grew faster than forecast in 2021-2023 (+3.6, +2.3, +2.4 points) and, outside 2020-21, the miss was larger when nominal growth beat the forecast (rho 0.74, p 0.043, n 8): inflation lifted revenue, but spending went further above plan. By category (next section), the 2021-2023 overruns are almost entirely **capital transfers (ESA D.9)**: +3.4, +3.1 and +4.2 points of GDP (94%, 61% and 118% of the miss of total expenditure; the next largest category is +0.3 in 2021 and 2023 and interest, +1.35, in 2022). This is consistent with the Eurostat decision of early 2023 under which Istat records the Superbonus and bonus facciate credits as payable tax credits, i.e. capital expenditure at the time of the works (reported by AGI on 1 March 2023, which also gives the revised deficits: 2021 9.0% against 7.2% before, 2022 8.0% against the 5.6% target), but D.9 also holds other capital transfers (investment grants among them), so the amount due to the building credits alone is not isolated here; a lower bound from the Superbonus alone is given in the next bullet. The 2020 miss splits between capital transfers (+1.5), social payments (+1.3) and subsidies (+0.6).
- **The Superbonus alone accounts for at least about half of the capital-transfer overrun of 2021-23.** ENEA's bulletins give Superbonus credits accrued of 11.7, 38.0 and 50.0 billion euro in 2021, 2022 and 2023 (investments for completed works 10.7, 34.5 and 45.8 billion; credit rate 1.10, 1.10, 1.09; 2021 is the cumulative since July 2020), against capital-transfer overruns of 62.0, 62.2 and 90.5 billion: 19%, 61% and 55% of the overrun, 46% over the three years (99.7 against 214.7 billion; 0.64, 1.90 and 2.34% of GDP against 3.37, 3.11 and 4.22). With the investment instead of the credit the shares are 17%, 56% and 51%. This is a **lower bound**: ENEA has the Superbonus only, so the facade bonus, the ordinary ecobonus and sismabonus and the other building bonuses are in the remainder (50.3, 24.2 and 40.4 billion), and so are other capital transfers; the overrun is measured against the plan, which may already hold part of the credits, and the snapshots are by transmission date of the asseverations, so late transmissions understate a year and overstate the next (the 2021 share is the lowest also because the Eurostat treatment books the credits when the works are done and 2021 is early in the programme). It makes the building-credit explanation of the overruns quantitatively plausible (more than half of the 2022 and 2023 overruns come from one measure), without separating it from the other bonuses.\n- **Istat's own figures put the Superbonus at about three quarters of the overrun, and with the facade bonus over four fifths.** The Superbonus effect on net borrowing of 0.9, 2.8 and 3.9% of GDP in 2021-23 is 16.5, 56.0 and 83.5 billion euro with the outturn GDP: 27%, 90% and 92% of the capital-transfer overrun of each year, 73% over the three years (156.0 against 214.7 billion); adding the facade bonus (about 23 billion, 2020-22, no annual split) gives 83% (179 against 215 billion). The ENEA credits are 64% of Istat's figure over the three years (71%, 68%, 60%): the gap is probably the super sismabonus (outside the SuperEcobonus reports) and the accounting rules for when and at what value a credit is booked, which this note cannot separate. What is left (about 36 billion euro, 17%) is other building bonuses, other capital transfers and the difference between the plan and the accounts. Source caveat: the Istat figures are quoted from a Camera dossier, not from the note itself; an earlier MEF estimate (DEF 2023) put the new treatment's effect at 0.2, 1.8 and 2.6 points of GDP in 2020-22 for the Superbonus and the facade bonus together, a different vintage.\n\n- **Three primary sources agree, and the year 2021 is where the building credits explain least.** The Bank of Italy's memorandum of 22 April 2024 puts the credits matured at about 1%, almost 3% and 3.7% of GDP in 2021-23 and almost 8% together, against Istat's 0.9, 2.8 and 3.9 (total 7.6): the same magnitudes from an official body, with the 2023 gap being Istat's later revision (3.7 preliminary of 5 April, 3.9 in the note of 26 April). Istat's hearing of March 2023 (the first booking in full) has the spending for the Superbonus and facade bonus together at about 2.0, 35.8 and 50.4 billion euro in 2020-22 (a chart without labels, +-0.5 billion), i.e. 58% of the 2021 overrun and 81% of 2022; later revisions raised 2022 (Superbonus alone 56.0 billion in the April 2024 vintage), so the vintages differ and are alternatives. Across the sources the building credits are 19% to 58% of the 2021 overrun, 61% to 96% of 2022 and 55% to 92% of 2023: **2021 is the year where at least 40% of the capital-transfer overrun (26 billion with Istat's March 2023 figure) is something else**. Istat's hearing also has the spending for tax credits in favour of all firms (about 18.8 billion in 2021 and 33.3 billion in 2022, 10.3 in 2020) and, in its Tavola 1, their composition by type: see the next bullet. The Istat note of 26 April 2024 itself (obtained from the Senate and read: 0.03, 0.9, 2.8, 3.9% of GDP, over 150 billion euro, over 175 with the facade bonus) confirms the Camera dossier's figures; it gives no split of the facade bonus by year.\n- **Firms' investment tax credits close most of the gap in 2021 and all of it in 2022.** Istat's Tavola 1 (hearing of 15 March 2023, page 23) splits the firms' tax credits by type; Istat does not give the ESA category of each, so the grouping is a judgment, in two widths: *core* = Transizione 4.0 and Investimenti Mezzogiorno (credits tied to the firm's investment, investment grants), *wide* = core + R&D, deferred-tax-asset conversion and the vehicle bonus. With the Istat March 2023 figures, the capital-transfer overrun of 2021 (62.0 billion) minus the Superbonus and facade spending (35.8) leaves 26.2 billion; the core credits are 7.0 billion (24.0% + 13.4% of 18.8) and the wide ones 13.6, leaving 19.2 (31% of the overrun) or 12.6 billion (20%) unexplained. In 2022 the overrun minus the building bonuses is 11.8 billion and the core credits alone are 12.4 (29.2% + 8.0% of 33.3): the overrun is fully explained (remainder -0.6, or -4.1 with the wide set). The subsidy-type credits (Covid, freight, audiovisual, fund-perduto, other, and the energy credits of 2022) are a cross-check on the category: 5.2 billion in 2021 against a subsidies overrun of 5.1 billion, and 17.4 billion in 2022 (energy credits 14.4) against 11.9. What is left in 2021 (13-19 billion, 20-31% of the overrun) is not identified: candidates are capital transfers outside tax credits (the PNRR and other grants, guarantees called) and the difference between the plan and the accounts. Limits: the vintages differ (March 2023 for the credits, the latest Eurostat vintage for the overrun), the chart totals are read to +-0.5 billion, the category grouping is a judgment, and the overrun is against the plan, which may already hold part of these credits.\n- **The unexplained part of 2021 sits in other capital transfers (D.99), not in investment grants (D.92).** Eurostat splits Italy's capital transfers: investment grants (D.92, where the building bonuses and the firms' investment credits are recorded) rise from 14.4 billion euro in 2019 to 58.8 in 2021 (+44.4), which the building bonuses plus the firms' investment credits (42.8 to 49.4 billion) account for; other capital transfers (D.99, cash grants) rise from 5.4 to 33.1 (+27.7), against 25.0 in 2020 (+19.6), the pandemic years. The part of the 2021 overrun left over by the credits (12.6 to 19.2 billion) is therefore inside the D.99 increase, which fits cash grants decided after the budget plan was written (the Sostegni decrees of 2021: the technical report of DL 73/2021 books the grants to firms as capital spending in net borrowing, which we saw in its prospetto) but cannot be tied to them item by item here. In 2022 the D.99 is back to 12.0 billion (+6.6) while D.92 is 75.5 billion above 2019, 9 to 13 billion more than the credits above, plausibly PNRR investment grants. These categories are Eurostat's (gov_10a_main, % of GDP times GDP), not a match to Istat's credit table.\n- **The Sostegni grants cover the 2021 D.99 increase, but authorized is not booked.** The technical reports and the law as converted (parliamentary dossiers of DL 41/2021 and DL 73/2021, Normattiva, `data/external/covid_grants_2021.csv`) authorize 11,150 million euro of contributi a fondo perduto in DL 41 (art. 1) and, in DL 73 (art. 1 as converted), 5,873 (automatic, down from 8,000 in the initial text), 3,400 (alternative) and 4,000 + 452.1 (worst-hit operators, of which 3,150 are unused DL 41 resources and only 850 new): 21.7 billion of distinct money. Other capital transfers (D.99) in 2021 are 33.1 billion: the 2019 level (5.4) plus 21.7 leaves 6.0 billion of other items. The Agenzia delle Entrate paid 5.21 billion of the automatic tranche (1.81 million beneficiaries) by 22 June 2021 against 5.87 authorized (89%); at that rate the grants would be 19.3 billion and the D.99 left over 8.4. The grants (19.3 to 21.7 billion) therefore cover the 12.6 to 19.2 billion that the building bonuses and firms' investment credits leave unexplained, in the order of magnitude, in both readings; the booked amounts are not known. This does not prove the classification: the technical reports book DL 41's grants as current spending, and the Government's table for DL 73 books them as capital spending in net borrowing; the D.99 increase and the 2020 jump of the same kind (+19.6 billion) point to Istat booking them as other capital transfers. Eurostat's April 2020 draft note on the COVID-19 measures discusses subsidies on production (D.39) and other capital transfers (D.99) for business support without settling it for lump-sum grants.\n- **Istat's accounts by receiving sector settle the 2021 question and replace the judgment grouping.** In the October 2022 edition the households' investment grants were 2.3 billion euro in 2021; the March 2023 edition, which books the Superbonus and the facade bonus as payable credits in full, has 37.9 (+35.6) and moves net lending of 2021 by -32.3 billion and of 2020 by -2.7 (exactly Istat's published revisions); the corporations' other capital transfers (D.99), where the cash grants of the pandemic decrees sit, were already 32.6 billion in 2021 in the October 2022 edition (so the grants were classified there from the start, not reclassified). In the latest edition (October 2026) the 2021 rise over 2019 of D.92 + D.99 is 73.2 billion: **36.9 (50%) households' investment grants** (the building bonuses), **7.6 (10%) corporations' investment grants** (the firms' investment credits), **27.0 (37%) corporations' other capital transfers** (the cash grants) and 1.8 other. The hearing's chart of the Superbonus and facade spending (2.0, 35.8 and 50.4 billion) matches the households' grants above 2019 of the March 2023 edition within 0.8 billion (2.0, 36.6, 50.1). Taking the households' rise above 2019 as Superbonus plus facade bonus, Istat's Superbonus effect of the note of 26 April 2024 leaves an implied **facade bonus of 1.5 (2020), 20.3 (2021), -0.1 (2022) and 0.6 (2023) billion**, 21.7 for 2020-22 (22.3 with 2023), which agrees with the 'about 23 billion' of the Camera dossier and answers the open question of the annual split (an assumption, with +-1 billion from the one-decimal Superbonus incidence). What this changes: the corporations' investment-grant rise of 7.6 billion in 2021 is close to the core credits of the grouping above (7.0) and the wide set (13.6) overstated it; in 2022 it is 19.4 against 12.4 (core) and 15.9 (wide), so more than the credits of Istat's table (probably the PNRR and other grants to firms); and the corporations' D.99 rise of 27.0 billion in 2021 exceeds the authorized Sostegni grants (21.7) by 5.3 billion, which the firms' tax credits of Istat's table (Covid credits 2.5 billion) and other transfers can cover but this cannot show. The 2021 remainder is therefore no longer open: it is the cash grants to firms, in Istat's own classification.\n- **Most of the 2021 deficit surprise is the March 2023 reclassification.** In the edition of October 2022 (before the Superbonus and the facade bonus were booked in full) the 2021 deficit was 128.9 billion euro, 7.01% of the latest GDP, against a target of 7.0: a surprise of +0.01 points. In the March 2023 edition it is 161.2 billion (8.77%, +1.77 points), in the latest 164.0 (8.92%, +1.92): the +1.9 points of the surprise table above are 1.77 points of accounting treatment and about 0.15 points of other revisions. For 2022 the first edition already includes the reclassification (+2.08 points in March 2023, +2.47 in the latest), so it cannot be cleaned in the same way; the 2022 plan (5.6) was also written under the cash treatment. This links to the cross-country ranking of notebook 41: Italy's 2021-23 overrun is largely an accounting overrun that other countries, which did not book payable credits this way, do not share, so the placement is a placement of the accounting regime as much as of the forecasts (the early-estimate check of notebook 41 shows the same: Italy's 2021 deficit revised up by 1.7 points of GDP after the early estimate, the others by 0.2 at most).\n- **With the first release of each year the misses look different, and most of the large revisions are 2021.** Re-running the decomposition of the deficit surprise with Istat's first edition of each year (2019-2025; the first edition containing the year, the March or April of the next one) instead of the latest: the 2021 surprise is -0.07 points (first release) against +1.92 (latest), a revision of +1.99 almost all in primary expenditure (+2.64, the building credits booked as capital transfers) partly offset by revenue (-0.66); 2022 +2.08 against +2.47 (first edition March 2023, already reclassified; revision +0.39: primary expenditure +0.70, revenue -0.24), 2023 +2.48 against +2.60, 2020 +7.18 against +7.17, 2019 -0.77 against -0.92, 2024 -0.88 against -0.91, 2025 -0.20 against -0.22. So the first-release outturns of 2022 and 2023 already show the overrun (+2.1 and +2.5 points, primary expenditure +3.0 and +3.7), and the 2021 overrun that stood out in the latest data was not visible at the first release; the correlations between the manovra and the surprise of the second family are computed on the latest vintage and the 2021 point is the one that moves most. The deficits are in % of the latest nominal GDP (the GDP of each edition is not in the account), so only the numerator changes and the denominator effect is the same in both vintages.\n- **The quarterly account shows when the credits were booked.** Other capital expenditure (D.92 + D.99) runs at 3.8 to 4.0 billion euro a quarter in 2019, rises with the pandemic grants (14.7 and 16.5 in the last two quarters of 2020, 28.1 in the second quarter of 2021 when the Sostegni grants were paid) and, with the credits, to 19.7-32.7 in 2022; **the fourth quarter of 2023 is 51.4 billion, 40% of the year (127.5)**, plausibly the last quarter before the rate fell to 70% for 2024 (Istat's note of 26 April 2024: 90% for 2023, 70% for 2024, 65% for 2025, with wide derogations in 2023), and **2024-Q1 falls to 7.6 billion**. The March 2023 booking moved the earlier quarters in an almost even way: net lending of 2021 by -9.1, -8.6, -6.8 and -7.8 billion over the four quarters (other capital expenditure +9.9, +9.5, +9.1, +8.8), of 2022 by -10.7, -15.0 and -21.4 billion in the first three quarters (the edition of January 2023 had only three quarters of 2022), so the credits accrue with the works through the year and the later years carry more. Quarterly figures are the raw series; the sum of the four quarters of 2021 (92.7 billion) matches the annual D.92 + D.99 (91.9) within 1%.\n- **The Superbonus went more to the richer regions, and the gap narrowed over time.** ENEA's investments for completed works by region (`data/external/enea_superbonus_regions.csv`, end 2023 and end 2025; SuperEcobonus only) give the North 53.9% of the end-2023 investment against 46.4% of the population (ratio 1.16; 1,851 euro per inhabitant), the Centre 18.2% against 19.9% (0.91; 1,454) and the South and Islands 27.9% against 33.7% (0.83; 1,319); by the end of 2025 the ratios are 1.15, 0.93 and 0.84 (2,283, 1,842 and 1,681 euro per inhabitant). Across the 20 regions the investment per inhabitant of the end-2023 snapshot goes with GDP per inhabitant (Spearman 0.51, permutation p 0.026; 0.43, p 0.063 at the end of 2025) and with the condominium share (0.45, p 0.049; 0.37, p 0.10 at the end of 2025). The highest per inhabitant are Valle d'Aosta (3,519 euro), Trentino-Alto Adige (2,462), Basilicata (2,331) and Abruzzo (2,313), the lowest Campania (1,070), Liguria (1,109), Sicilia (1,137) and Puglia (1,238): the relation is moderate with clear exceptions in both directions (Basilicata and Abruzzo high, Lazio, Liguria and Campania low). Twenty regions, one pre-specified test (end-2023 snapshot, per inhabitant) and robustness checks that are not a family; GDP per inhabitant is workplace based and not the income of the households that used the credit; the investments follow the housing stock, ownership and the ability to advance the cost, so this is a description of who benefited, not an explanation. Istat's regional accounts (esploradati, households' disposable income by region) stop at the income account: they have no capital transfers received by region, so the regional split of the households' investment grants cannot be taken from them.\n- **The income gradient of the Superbonus appeared in 2023, not before.** The ENEA monthly reports (62 month ends, `data/external/enea_superbonus_monthly.csv`) split the new admitted investment by window. The South's ratio of its share of the new works to its population share is 1.02 up to August 2021, 0.90 in September-December 2021, 1.03 in the first half of 2022 and 0.87 in the second, then **0.69 in the first half of 2023**, 0.84, 0.72 in 2024 and 0.94 in 2025-26 (the North 0.96, 1.09, 0.98, 1.10, 1.28, 1.16, 1.25, 1.04). In the pre-specified family of 8 windows (Spearman across the 20 regions of the admitted investment per inhabitant against GDP per inhabitant) the correlation is -0.17, 0.16, 0.00 and 0.31 up to the end of 2022 (none significant), then 0.74 in the first half of 2023 (q 0.003), 0.42 in the second half (q 0.18), 0.59 in 2024 (q 0.031) and -0.17 in 2025-26: **2 of 8 windows survive false-discovery control** (the completed-works version agrees: 0.68, q 0.013, and 0.58, q 0.041), and the placebo that shuffles the regional incomes gives a discovery in 5.2% of 600 shuffles (mean 0.11 windows). The cumulative gradient goes from -0.19 in August 2021 to 0.19 in November 2022, 0.29 in December, 0.40 in January 2023 and 0.45 in March 2023, then stays at 0.40-0.48 (0.40 in December 2023, 0.48 in December 2024, 0.40 in September 2026): it was built in the winter of 2022-23, the months of the 2023 rate cut and of the stop to credit transfers (known policy changes that this note does not date or test). New admitted investment peaks at 8.2 billion euro in September 2022 (46.3 billion in 2022, 40.2 in 2023, 16.4 in 2024, 4.9 in 2025); the South also completes more slowly (87% of the admitted investment completed at the end of 2023 against 90% in the North; 97.6% against 98.8% in September 2026). Limits: 20 regions per window, one family and a shuffled-income placebo, the SuperEcobonus only, flows by transmission date (late transmissions move works between windows and ENEA's own revisions give a fall of 602 million in May 2026, 3,051 buildings), and GDP per inhabitant is workplace based; this is who used the credit when, not why. Two basis notes: the annual report's regional table (works ended by 31 December, later extraction) is 0.75 to 1.05 times the monthly report of the same month end by region (Valle d'Aosta lowest, Molise highest) with the same ranking (Spearman 0.99), so the end-2023 test above is not sensitive to the basis; the monthly national totals match the transcribed snapshots at the bulletin basis (December 2023, March 2025) and are 5% and 3% above the annual-report figures of December 2021 and 2022 (not re-based in the lower bound above).\n- **Istat's account and the Eurostat figures used here agree.** Putting Istat's latest edition in % of the project's nominal GDP and comparing with Eurostat's % of GDP for Italy over 2005-2024: net lending, compensation of employees, intermediate consumption, interest, subsidies and capital transfers (D.92 + D.99) differ by at most 0.04 to 0.07 points of GDP in any year (the rounding of Eurostat's one decimal is 0.05), total expenditure by up to 0.19 (2021) and total revenue by up to 0.13 (2018), both slightly higher in Istat in 2010-2021 (mean +0.1), which is probably the GDP vintage and not a definition difference. So the Eurostat numbers used in notebooks 38 and 41 are the same accounts as the Istat ones, and the Istat editions add the revision history, the receiving sector and the quarters.\n- **Limits.** About ten years, so only a rank correlation of about 0.65 or more can be told from zero: a moderate true link would not be detected. The outcomes are the year-on-year change in the outturn (latest Eurostat vintage), not the manovra's effect; the cycle, interest and one-off items move them too. The deficit target is the one stated when the law was proposed (for 2019 also the revised one); targets for 2016-2022 come from the draft budgetary plans (two tables of each agree) and for 2023-2026 from the dossiers' programmatic row. Figures are ex-ante estimates from the government's technical reports for the bill as first presented (2022: rebuilt from the measure lines of the budget service's note, not a published table), one country and no placebo countries (a placebo ranking would need the same tables for other euro-area countries). Descriptive associations only.
- **No link between the regional use of the Superbonus and the change of the regional outcomes.** The pre-specified test (investment per inhabitant against the change of the employment rate 20-64, 2019-23, 20 regions) gives rho -0.31 (permutation p 0.18, within zones 0.21); in the 2015-19 placebo window the same correlation is +0.37 (p 0.11), so the regions that used the credit more were not the ones whose employment was falling behind. Of the 12 tests (6 outcomes x 2 periods) none survives BH (smallest q 0.27: unemployment change 2019-23, rho +0.49, p 0.029; female employment 2015-19, rho +0.46, p 0.044); the shuffled family gives a discovery in 3.8% of 600 shuffles (mean 0.05). Per euro of 2019 GDP rho is 0.30 (p 0.19) for the primary outcome; beyond the 2021 income level it is -0.08 and beyond income and the 2015-19 change 0.08. A negative result with 20 regions: a gap of the size seen in Italy's recovery would need a larger sample or the project-level data; this does not say the credit had no regional effect. Descriptive.\n- **The timing cannot attribute the 2023 income gradient to the winter rules.** Dates (two sources each: Normattiva or the Gazzetta, the Camera dossier, ANCE, Assolombarda): DL 176/2022 published 18 November 2022 with the CILA cut-off at 25 November, budget law 29 December, DL 11/2023 published 16 February and in force 17 February 2023, conversion 11 April; the 30 March 2023 MEF note is `partly verified`. The monthly gradient of the new admitted investment (rho over 20 regions) is -0.28 in September 2022, 0.32 in October (before the decree), 0.55 in November, 0.71, 0.78, 0.80 in December-February, then 0.66, 0.32, 0.29, 0.54: it begins to rise a month before the first rule and is carried by the single-family houses, whose flow falls from 5.1 billion euro in September 2022 (the 30 September condition date, rho -0.66) to 1.4, 0.8 and 1.3 billion in October-December; the condominiums have a positive gradient throughout. Pre-specified contrasts (2 events x 2 measures): none stands out among the possible cut months (the December 2022 gradient contrast +0.42 ranks 12 of 50, p 0.24; BH q 0.77 for all four). Reads: the winter rules cannot be shown to have created the gradient, and the data cannot date their effect either (flows by transmission lag the start of works). Descriptive.\n- **Re-basing the lower bound on the monthly series moves credits between years, not the total.** Credits accrued of 12.3, 39.0 and 48.4 billion euro in 2021-23 against 11.7, 38.0 and 50.0 on the annual-report basis (ratios 1.05, 1.03, 0.97), so the share of the capital-transfer overrun is 19.8%, 62.7% and 53.5% (19%, 61%, 55% before); over the three years 99.7 billion and 46.5% on both bases, because the December 2023 cumulative is the same. Late transmissions raise the earlier years and lower the last one, as the basis note said.
"""))

NOTEBOOKS["39_contract_timing"] = [
    (M, """# Contract timing and the real wage loss after the 2021-23 inflation shock
Notebook 36 compared countries: collective-bargaining coverage and union density against the pass-through of inflation to wages (21-23 countries, nothing found). Here the question moves inside Italy, where **collective agreements are renewed with long and uneven delays**, and the data give many more observations: Istat's index of **contractual hourly wages** for 113 agreement groups (`155_318_DF_DCSC_RETRCONTR1C_4`, base December 2021 = 100, monthly since 2005) and its **bargaining-tension indicators** (share of employees whose contract has expired and the months they have waited, for six aggregates, `149_319_DF_DCSC_RETRCONTR1T`).

**Design.** Contractual wages move when an agreement's increment takes effect, so each group's index is a staircase. For every leaf group (no sub-group: 67 private, 13 public; 78 with data over the window) at December 2021, the eve of the shock: **staleness** = months since the last increment (a month-on-month rise of at least 0.25% of the index) and **standstill** = months without an increment among the 24 before. Outcome: the **real change** of the contractual wage between December 2021 and December 2024 (and 2025), the index change deflated by Italy's cumulative December-on-December HICP inflation (4.2%, 12.2%, 0.6%, 1.3% and 1.2% in 2021-2025: 14.3% to 2024, 15.7% to 2025). The question, exploratory and two-sided: do groups whose agreement was already old when the shock began end up with a different real loss? Spearman rank correlation with a permutation p, BH over the pre-specified family (two predictors, two horizons), leave-one-group-out, a calm-period placebo (December 2016 to 2019) and a shuffled-outcome null.

**A change of unit, declared.** The sub-sectors under one parent group (for instance the manufacturing branches) share the same national agreements, so their staircases are not 78 independent observations. The first test was run on the 78 leaves, and its p-value turned out to depend on the null used (0.002 shuffling within the private and public branch, 0.5 shuffling within parent groups). The **primary analysis is therefore on the 30 parent groups** (means of their sub-sectors; the permutation shuffles within the private and public branch); the leaf-level tests are kept as sensitivity and are reported with their caveat. This choice was made after seeing the leaf-level results, not before.

**Limits of the data.** Contractual wages exclude wage drift, bonuses outside the agreements and one-off payments; the groups are Ateco sectors, not contracts, and carry no employee weights; the number of workers per agreement is not in the flows, and renewal dates come from CNEL's archive (section E) for 74 of the 78 leaves through a hand-made mapping. Descriptive associations only."""),
    (C, '''from cifresignificative.analysis import contracts as CT, experiments as ex
from cifresignificative.analysis.hf import load_highfreq
cres = ex.run_contracts_all()
leaves, clu = cres["c1_leaf_groups"].set_index("group"), cres["c12_clusters"].set_index("group")
dec = CT.december_inflation(load_highfreq())
print("December-on-December HICP inflation, Italy (%):", dec.loc[2016:2025].round(1).to_dict(), "| cumulative 2021-24:", round(CT.cumulative_inflation(dec, 2021, 2024), 1), "| 2021-25:", round(CT.cumulative_inflation(dec, 2021, 2025), 1))
print("leaf groups with data:", len(leaves), leaves.branch.value_counts().to_dict(), "| parent groups:", len(clu), "with", clu.n_leaves.value_counts().sort_index().to_dict(), "sub-sectors each")
print("leaf groups, medians by branch (nominal and real change of the contractual wage Dec 2021 - Dec 2024, %; staleness and standstill in months at Dec 2021):"); display(leaves.groupby("branch")[["nominal_change", "real_change", "staleness_months", "standstill_24m"]].median().round(2))
print("share of leaf groups with a real loss by Dec 2024:", round(float((leaves.real_change < 0).mean()), 3), "| by Dec 2025:", round(float((leaves.real_change_25 < 0).mean()), 3), "| range of real change to 2024:", round(float(leaves.real_change.min()), 1), "to", round(float(leaves.real_change.max()), 1))
print("aggregates (Istat's own aggregates; nominal and real change, %):"); display(cres["c10_aggregates"].set_index(["end", "group"])[["name_en", "staleness_months", "standstill_24m", "nominal_change", "real_change"]].round(2))'''),
    (C, '''cols = ["x", "y", "n", "rho", "p", "q_value", "loo_min", "loo_max"]
print("PRIMARY - 30 parent groups (x shuffled within the private and public branch; BH over the four tests):"); display(cres["c2_cluster_family"][cols].round(3))
print("private parent groups only:"); display(cres["c5_cluster_private"][cols].round(3))
print("step threshold (0.15%, 0.25%, 0.5%) - 2024 outcome:"); display(cres["c6_cluster_thresholds"][["step", "x", "n", "rho", "p"]].round(3))
print("SENSITIVITY - 78 leaf groups, x shuffled within branch (anti-conservative: sub-sectors share contracts):"); display(cres["c3_leaf_family_branch"][cols].round(3))
print("SENSITIVITY - 78 leaf groups, x shuffled within parent group (conservative: removes the between-parent variation):"); display(cres["c4_leaf_family_within_parent"][cols].round(3))
print("PLACEBO - the same test for December 2016 to December 2019 (cumulative inflation 2.6%):"); display(cres["c7_placebo_2016_19"][cols].round(3))
nl = cres["c8_null"]; print("shuffled-outcome null,", len(nl), "shuffles: mean discoveries", round(float(nl.discoveries.mean()), 3), "| shuffles with a discovery", round(float((nl.discoveries > 0).mean()), 3), "| share of p < 0.05", round(float(nl.share_p_below_005.mean()), 3))
print("mechanism check on the parent groups (steps_after = months with an increment between Dec 2021 and Dec 2024):"); display(cres["c11_persistence"].round(3))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ax, x, lab in ((axes[0], "staleness_months", "months since the last increment, Dec 2021"), (axes[1], "standstill_24m", "months without an increment among the 24 before Dec 2021")):
    for br, colr in (("private", "#2a78d6"), ("public", "#d94f3d")):
        s = clu[clu.branch == br]; ax.scatter(s[x], s.real_change, s=18 + 14 * s.n_leaves, color=colr, alpha=.7, label=br)
    ax.axhline(0, color="k", lw=.5); ax.set_xlabel(lab); ax.set_ylabel("real change of the contractual wage, Dec 2021 - Dec 2024, %")
axes[0].legend(fontsize=7); fig.suptitle("Parent groups of agreements (marker size = number of sub-sectors)", fontsize=9); plt.tight_layout(); plt.show()'''),
    (C, '''tn = cres["c9_tension"].copy(); tn.columns = [c if isinstance(c, str) else "|".join(x for x in c if x) for c in tn.columns]
sel = ["year", "base"] + [f"{k}|{s}" for k in ("employees_expired_pct", "months_all") for s in ("private sector", "public administration", "total economy")]
print("bargaining tension at December of each year (base 2015 up to 2020, base 2021 from 2021; not spliced): share of employees with an expired contract (%), months of waiting averaged over all employees:"); display(tn[sel].round(1))
tt = tn.set_index("year")
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for s, colr in (("private sector", "#2a78d6"), ("total economy", "#444444")): ax.plot(tt.index, tt[f"months_all|{s}"], "o-", color=colr, label=f"{s}, months waited by all employees")
ax2 = ax.twinx(); ax2.bar(dec.loc[2015:2025].index, dec.loc[2015:2025].values, color="#d94f3d", alpha=.25, width=.6); ax2.set_ylabel("December HICP inflation, % (bars)")
ax.set_ylabel("months"); ax.legend(fontsize=7, loc="upper left"); ax.set_title("Waiting time for contract renewals and inflation", fontsize=9); plt.tight_layout(); plt.show()'''),
]

NOTEBOOKS["39_contract_timing"] += [
    (M, """## E. Renewal dates from CNEL's archive, and a sector deflator
The staircase proxy above infers renewals from the index. Two follow-ups, **declared before the results were seen**.

**(1) Renewal dates.** CNEL's open-data archive (`sources/cnel_contracts.py`, map in `data/external/cnel_istat_map.csv`, provenance and limits in `data/external/source/cnel/README.md`) gives the signing date and the contractual expiry of every renewal. Each Istat leaf group was mapped by hand to one CNEL contract (74 of 78 leaves; four groups with no collective contract: police, armed forces, fire brigade, the Presidency of the Council). The mapping labels are my reading of titles and signatories, not an independent second source. Cycles signed by minor unions alone are dropped when the contract has confederal (CGIL, CISL, UIL) cycles. **Primary test:** the parent groups of the `verified` leaves; x = months since the last signed renewal and months the contract had been expired at December 2021; y = the real change to 2024 and to 2025; x shuffled within the private and public branch; BH over the four tests; the staircase proxy is re-run on exactly the same groups. **Sensitivities:** the leaves, the parents with the `partly verified` leaves added, the 2016-2019 placebo and a shuffled-outcome null.

**(2) Sector deflator.** Goods HICP for agriculture, industry and construction, services HICP for private services, all-items for the public sector. Consumer prices are not the prices of a branch's output: this is a sensitivity check of the national deflator, not a better measure of the loss. Replacing the national deflator by another national one cannot change a rank correlation, so only a deflator that differs by sector matters. As a **composition check**, the national deflator is kept and x is shuffled within the sector (goods, services, public) instead of within the branch."""),
    (C, """from cifresignificative.sources import cnel_contracts as CNm, istat_contracts as ICm
tm = cres["c13_renewal_leaves"].set_index("group")
print("leaf groups mapped to a CNEL contract:", len(tm), tm.evidence.value_counts().to_dict(), "| distinct contracts:", tm.cnel_code.nunique(), "| renewal signed after Dec 2021:", int(tm.renewed_after.sum()), "of", len(tm))
print("timing at December 2021, medians over the mapped leaves (months):"); display(tm[["months_since_renewal", "months_expired", "months_to_renewal"]].median().round(1).to_frame("median").T)
print("agreement of the staircase proxy with the dates (Spearman, mapped leaves):"); display(cres["c18_proxy_agreement"].round(3))
cols = ["x", "y", "n", "rho", "p", "q_value", "loo_min", "loo_max"]
print("PRIMARY (renewal dates) - parent groups of the verified leaves, x shuffled within branch, BH over four:"); display(cres["c14_cnel_cluster_family"][cols].round(3))
print("the staircase proxy on the same groups:"); display(cres["c15_proxy_same_clusters"][cols].round(3))
print("SENSITIVITY - verified leaves (branch shuffle, anti-conservative):"); display(cres["c16_cnel_leaf_family"][cols].round(3))
print("SENSITIVITY - parents with the partly verified leaves added (renewal dates and proxy, BH over eight):"); display(cres["c17_cnel_cluster_with_partly"][cols].round(3))
print("PLACEBO - renewal dates at December 2016, real change to December 2019:"); display(cres["c19_cnel_placebo_2016_19"][cols].round(3))
nl = cres["c20_cnel_null"]; print("shuffled-outcome null,", len(nl), "shuffles: mean discoveries", round(float(nl.discoveries.mean()), 3), "| share of p < 0.05", round(float(nl.share_p_below_005.mean()), 3))
print("cumulative HICP by component, Dec 2021 to Dec 2024 / 2025 (%):"); display(cres["c23_deflators"].pivot(index="component", columns="end", values="cumulative_pct").round(1))
print("SECTOR DEFLATOR - staircase proxy, 30 parents:"); display(cres["c21_sector_deflator_proxy"][cols].round(3))
print("SECTOR DEFLATOR - renewal dates, verified parents:"); display(cres["c22_sector_deflator_cnel"][cols].round(3))
print("COMPOSITION - national deflator, x shuffled within sector, staircase proxy (30 parents):"); display(cres["c24_proxy_within_sector"][cols].round(3))
print("COMPOSITION - national deflator, x shuffled within sector, renewal dates:"); display(cres["c25_cnel_within_sector"][cols].round(3))
print("mean timing and national-deflator real change by sector of the parent groups:"); display(cres["c26_timing_by_sector"].round(2))
defl = CT.deflator_of(ICm.load_groups()); sec = defl.reindex(leaves.index).groupby(leaves.parent).first()
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2)); colr = {"goods": "#2a78d6", "services": "#1f9d6b", "all-items": "#d94f3d"}
cl_cnel = cres["c14_cnel_cluster_family"]
for ax, x, lab, d in ((axes[0], "staleness_months", "months since the last increment, Dec 2021", clu), (axes[1], "months_expired", "months the contract had been expired, Dec 2021 (CNEL dates)", None)):
    if d is None:
        t = tm[tm.evidence == "verified"].join(leaves[["parent", "real_change"]]); d = t.groupby("parent").agg(months_expired=("months_expired", "mean"), real_change=("real_change", "mean"))
    for k, c in colr.items():
        s = d[sec.reindex(d.index) == k]; ax.scatter(s[x], s.real_change, s=26, color=c, alpha=.75, label={"all-items": "public"}.get(k, k))
    ax.axhline(0, color="k", lw=.5); ax.set_xlabel(lab); ax.set_ylabel("real change, Dec 2021 - Dec 2024, national HICP, %")
axes[0].legend(fontsize=7); fig.suptitle("Parent groups by sector: the old agreements are services and public ones", fontsize=9); plt.tight_layout(); plt.show()"""),
]

NOTEBOOKS["39_contract_timing"] += [
    (M, """## F. Weighting the groups by employees
The tests above give every group the same weight, so a small sector counts as much as the 2.8 million employees of private services. CNEL and INPS publish the number of employees that declare each agreement code (December 2025, UNIEMENS; `sources/cnel_workers.py`, three Excel files kept in `data/external/source/cnel/workers/`). **Declared before the results were seen:** a leaf group gets the count of its mapped contract, split equally among the leaves that share the code; a parent group gets the sum of its leaves; the weighted statistic is the weighted Pearson correlation of the ranks (weighted Spearman) with the same within-branch permutation p (x shuffled, the weight stays with the group), BH over the four tests of the proxy family, and a shuffled-outcome null. Limits: the counts are a **2025 snapshot** used as relative weights for a 2021-24 outcome; the **public agreements (ARAN) have no count**, so every weighted test is a test on the private groups (26 of 30 parents, 64 of 78 leaves); the weighting shrinks the effective sample (Kish n_eff = (sum w)^2 / sum w^2), which is reported next to n, so a weaker p is partly lost power, not necessarily a weaker association."""),
    (C, """w = cres["c31_weights"].set_index("group").sort_values("employees", ascending=False)
print("parents with a count:", int(w.employees.notna().sum()), "of", len(w), "| employees covered:", int(w.employees.sum()), "| the three largest hold", round(float(w.employees.head(3).sum() / w.employees.sum()), 3))
display(w.head(6).round(2))
print("employee-weighted against unweighted real change (leaf groups with a count):"); display(cres["c27_weighted_summary"].round(3))
cols = ["x", "y", "n", "n_eff", "rho", "rho_weighted", "p", "q_value"]
print("WEIGHTED proxy family - private parent groups, BH over four:"); display(cres["c28_weighted_proxy"][cols].round(3))
print("the same rows, unweighted (n_eff = n):"); display(cres["c29_unweighted_same_rows"][["x", "y", "n", "rho", "p", "q_value"]].round(3))
print("WEIGHTED renewal-date family - parents of the verified leaves with a count:"); display(cres["c30_weighted_cnel"][cols].round(3))
print("WEIGHTED proxy on the leaves (branch shuffle, anti-conservative):"); display(cres["c32_weighted_leaf_proxy"][cols].round(3))
nl = cres["c33_weighted_null"]; print("shuffled-outcome null of the weighted family,", len(nl), "shuffles: mean discoveries", round(float(nl.discoveries.mean()), 3), "| share of p < 0.05", round(float(nl.share_p_below_005.mean()), 3))"""),
]

NOTEBOOKS["39_contract_timing"] += [
    (M, """### F2. Adding the public agreements
The public groups have no CNEL-INPS count. The RGS **Conto annuale** (`sources/conto_annuale.py`; table OCCUPAZIONE, year 2021, the start of the wage window, 3.24 million heads over 47 contracts, downloaded by script) counts staff by contract, so each public leaf group gets the heads of its contract (map `data/external/conto_annuale_istat_map.csv`: ministries MNST, tax agencies AGFI, public bodies EPNE, regions and local authorities RALN, health service SSNA, research ERIC, music academies AFAM, schools SCLA, university staff UNIV, Presidency of the Council PCMD, fire brigade VVFF; police and armed forces are the sums of the contracts of the Conto annuale areas *Corpi di polizia* and *Forze armate*). The police and armed-forces groups have no CNEL contract and so no renewal date, but the staircase proxy does not need one, so they enter the proxy tests. The weights now mix a 2025 private snapshot with 2021 public heads (both heads of staff, different years: declared). Same statistics as in F, all 77 groups with a weight (the Presidency's and the others' parents included). **Note:** a parent's real change is the unweighted mean of its leaves, so the public parents (Z2560 holds ministries, health, regions and the Presidency, +5.6% for the latter) blend very different leaves; the leaf-level weighted test does not have this issue and is shown as well."""),
    (C, """wa = cres["c35_weights_all"].set_index("group").sort_values("employees", ascending=False)
print("parents with a weight:", int(wa.employees.notna().sum()), "of", len(wa), "| employees covered:", int(wa.employees.sum()), "| public share:", round(float(wa[wa.branch == "public"].employees.sum() / wa.employees.sum()), 3), "| the three largest hold", round(float(wa.employees.head(3).sum() / wa.employees.sum()), 3))
display(wa.head(6).round(2))
print("employee-weighted against unweighted real change, private and public leaf groups:"); display(cres["c34_weighted_summary_all"].round(3))
cols = ["x", "y", "n", "n_eff", "rho", "rho_weighted", "p", "q_value"]
print("WEIGHTED proxy family - all 30 parents, BH over four:"); display(cres["c36_weighted_proxy_all"][cols].round(3))
print("the same rows, unweighted:"); display(cres["c37_unweighted_same_rows_all"][["x", "y", "n", "rho", "p", "q_value"]].round(3))
print("WEIGHTED proxy on the 77 leaves (branch shuffle, anti-conservative):"); display(cres["c39_weighted_leaf_proxy_all"][cols].round(3))
print("WEIGHTED renewal-date family, all verified parents:"); display(cres["c38_weighted_cnel_all"][cols].round(3))
nl = cres["c40_weighted_null_all"]; print("shuffled-outcome null of the weighted family,", len(nl), "shuffles: mean discoveries", round(float(nl.discoveries.mean()), 3), "| share of p < 0.05", round(float(nl.share_p_below_005.mean()), 3))"""),
]

NOTEBOOKS["39_contract_timing"] += [
    (M, """### F3. Weights inside a parent, a second year and the special-statute contracts
F2 left three loose ends, all **chosen after F2 was seen**, so they are alternative specifications of one question, not a new test family: 8 variants are shown and none is singled out in advance. **(1) Weights inside a parent.** In F and F2 a parent's real change and timing were the *unweighted* means of its leaves while its weight was the sum of the leaves' employees, so a parent blending a large and a small leaf (the Presidency of the Council with the health service) was inconsistent. `weighted_cluster_table` averages the leaves by employees. **(2) Weighting year.** The Conto annuale of 31 December **2024** (the end of the wage window) next to 2021 (the start); the private weights stay the 2025 counts. **(3) The special-statute regions and autonomous provinces, and the Bolzano and Trento school contracts** (own contracts: REFR, RESI, RESA, REVA, RETN, PRTN, PRBZ, SCBZ, SCTN) added to the regional and school groups. Same statistics as in F, BH within each variant and family."""),
    (C, """v = cres["c41_weighted_variants"]
pw_ = cres["c42_parent_weighted_means"].set_index("group")
print("parent real change, equal-weighted mean of the leaves against the employee-weighted mean (2021 weights), the groups that move most:"); display(pw_.assign(shift=pw_.real_change - pw_.real_change_unweighted_mean).sort_values("shift", key=abs, ascending=False).head(6)[["branch", "n_leaves", "employees", "real_change_unweighted_mean", "real_change", "shift"]].round(2))
c = ["year", "extended", "parent_mean", "x", "y", "n", "n_eff", "rho", "rho_weighted", "p", "q_value"]
print("PRIMARY of this section - proxy, parents with employee-weighted means, 2021 heads, BH over four:"); display(v[(v.level == "parent") & (v.family == "proxy") & (v.parent_mean == "weighted") & (v.year == 2021) & ~v.extended][c[3:]].round(3))
def pick(d, x, y):
    t = d[(d.x == x) & (d.y == y)].set_index(["parent_mean", "year", "extended"]) if len(d) else d
    return t
summary = pd.concat({"staleness, to 2024": pick(v[(v.level == "parent") & (v.family == "proxy")], "staleness_months", "real_change"), "standstill, to 2024": pick(v[(v.level == "parent") & (v.family == "proxy")], "standstill_24m", "real_change"),
                     "months expired, to 2024 (24 parents)": pick(v[(v.level == "parent") & (v.family == "cnel")], "months_expired", "real_change")})[["n_eff", "rho", "rho_weighted", "q_value"]]
print("all variants (weighted Spearman and its q; 'unweighted' = equal-weighted parent means as in F2):"); display(summary.round(3))
lf = v[(v.level == "leaf") & (v.x == "standstill_24m") & (v.y == "real_change")].set_index(["year", "extended"])[["n_eff", "rho", "rho_weighted", "q_value"]]
print("77 leaves, standstill to 2024 (branch shuffle, anti-conservative):"); display(lf.round(3))"""),
]

NOTEBOOKS["39_contract_timing"].append((M, """## Takeaways (Istat contractual wages by agreement group, December 2021 to December 2024/25; exploratory and descriptive)
- **Almost every agreement group lost real contractual wages in the shock.** Of the 78 leaf groups, 69 (88.5%) had a lower real contractual wage in December 2024 than in December 2021 and 67 (85.9%) in December 2025; cumulative inflation was 14.3% to 2024 and 15.7% to 2025 (December-on-December: 4.2%, 12.2%, 0.6%, 1.3%, 1.2% in 2021-2025). The median group gained 7.5% (private) and 7.3% (public) in nominal terms and lost 6.0% and 6.2% in real terms by 2024; the range is -12.5% to +5.6%. Istat's own aggregates lost less (industry -2.7%, agriculture -4.2%, private sector -4.9%, total economy -5.4% by 2024; -1.7%, -3.7%, -3.6%, -3.8% by 2025) and private services most (-7.0% and -5.3%).
- **Groups whose agreement was already old when the shock began lost more.** On the 30 parent groups, the months since the last increment at December 2021 against the real change to December 2024: rho -0.58 (to 2025: -0.55); the months without an increment among the 24 before: -0.52 (-0.51). All four tests have p about 0.001 and q 0.001 (permutation, shuffling within the private and public branch), the leave-one-group-out range of rho is -0.47 to -0.64, and the result holds for the 26 private parent groups alone (-0.60 and -0.58) and with step thresholds of 0.15% and 0.5% instead of 0.25% (staleness -0.59 and -0.50, standstill -0.53 and -0.48). Median staleness at December 2021 was 9 months for the private leaf groups and 29 for the public ones. This is the association across all parent groups; section E shows that a large part of it is a difference between sectors.
- **The check that should be flat is flat.** The same test for December 2016 to December 2019, when inflation was 2.6% in total, gives rho -0.06 (staleness) and +0.06 (standstill), p 0.87; the shuffled-outcome null gives a discovery in 4.5% of 200 shuffles (nominal about 5%).
- **The association is between parent groups, not within them.** On the 78 leaves with the branch shuffle, standstill gives -0.32 (p 0.002) and staleness -0.17 (p 0.12); shuffling within parent groups removes the between-parent variation and gives p 0.5-0.83. Sub-sectors under one parent share the national agreements and move together, so the leaf-level p-values overstate the evidence and the effective sample is the 30 parent groups; the unit was changed after the leaf-level results were seen.
- **A timing channel is consistent with the data, not shown by them.** Parent groups with an older last increment also had fewer increments in the following three years (rho -0.40 between staleness and the number of months with an increment after December 2021; -0.18 for standstill), and groups with more increments afterwards lost less (rho 0.45). That fits slow renewal of expired agreements; it does not exclude that sectors with weaker bargaining power both renew late and lose more.
- **Waiting time moved with the shock and then collapsed.** Averaged over all private-sector employees, the wait for a renewal went from 11.1 months (December 2021) to 13.1 (2022) and 16.6 (2023), then fell to 4.1 (2024) and 5.5 (2025) with the wave of renewals; the share of private employees with an expired contract was 39.8%, 35.0%, 38.0%, 37.1% and 26.1% over 2021-2025. In public administration all employees have an expired contract at almost every December (0.1% in 2018) with waits of 36.1, 12.0, 24.0, 36.0 and 16.8 months over 2021-2025: renewals come late and in bunches. The base-2015 and base-2021 series are not spliced.
- **Contract-level renewal dates give a partial confirmation, on a smaller sample.** For 53 leaves with a `verified` CNEL mapping (24 parent groups) the months the contract had been expired at December 2021 go with a larger real loss to 2024 (rho -0.54, p 0.005, q 0.022, the only one of four tests that survives), the months since the last signed renewal less so (-0.36, q 0.17), and nothing survives against the 2025 outcome (-0.28 and -0.03). The staircase proxy on the same 24 groups gives -0.36 for both measures (q 0.18): the -0.58 of the 30 parents does not carry over to this subset, and the proxy agrees only moderately with the dates (rho 0.44 for staleness against the months since renewal, 0.64 for the standstill against the months expired, n 73). On the 53 leaves with the branch shuffle the expiry gives -0.21 (q 0.29). Adding the 21 `partly verified` leaves (29 parent groups) strengthens everything (expiry -0.61, q 0.002; months since renewal -0.48, q 0.011; staleness -0.62), but that mixes evidence levels. The 2016-2019 placebo is not flat for the expiry (rho -0.39, p 0.084, q 0.17, n 21; months since renewal -0.19): a tendency for groups with an expired contract to lose real wages exists in calm years too. The shuffled-outcome null gives 0.07 discoveries per shuffle and 4.9% of p below 0.05.
- **Much of the association is a difference between sectors.** The old agreements are in services and in the public sector: mean staleness at December 2021 is 9.9 months for the 15 parent groups of agriculture and industry (goods), 34.1 for the 11 of private services and 28.2 for the 4 public ones; the months of expiry are 0.5, 22.5 and 36.0; the real change with the national deflator is -3.9%, -7.5% and -6.6%. Two checks follow. **(a) Sector deflator** (goods HICP 16.0% to 2024, services 11.3%, all-items 14.3%): the staircase proxy against the 2024 outcome falls to rho -0.14 (q 0.45) and -0.25 (q 0.26), against 2025 it stays at -0.41 and -0.44 (q 0.047 and 0.029), and the renewal dates give -0.09 to 0.11 with q 0.95. **(b) National deflator, x shuffled within sector:** rho is unchanged (-0.51 to -0.58) but the permutation p rises to 0.016-0.075 and none of the four tests survives (q 0.063-0.075); with the renewal dates the expiry gives -0.54 with q 0.34. So the data do not separate the age of an agreement from the sector it belongs to: within sectors there are only three strata, and the goods groups are almost all unexpired in December 2021 (0.5 months on average), so little variation is left there. Consumer prices are not a branch's output prices, so (a) is a sensitivity, not a better measure.
- **Weighting by employees keeps the sign and size but loses the significance, because a few agreements dominate.** CNEL-INPS counts (2025) exist for 64 of 78 leaves and 26 of 30 parents (10.2 million employees, private only; the public agreements have no count); the three largest parents hold 50.3% of them (private services 2.75 million). Weighted by employees the mean real change is -5.75% (unweighted -5.87%) and 81.8% of employees were in a group with a loss (87.5% of the groups). The weighted rank correlation of the proxy with the real change is -0.69 (staleness) and -0.57 (standstill) to 2024, -0.71 and -0.61 to 2025, against -0.60, -0.58, -0.63 and -0.65 unweighted on the same 26 parents, so the association is not carried by small sectors. The effective sample is only 8.3 groups: p 0.024 and 0.020 for staleness (BH q 0.048, the two tests that pass), 0.098 and 0.064 for standstill (q 0.098 and 0.085); the shuffled-outcome null of the weighted family is calibrated (0.015 discoveries per shuffle, 3.1% of p below 0.05). The renewal-date family does not hold once weighted (n_eff 6.1; months expired -0.50 and -0.38, q 0.87; months since renewal 0.00 and +0.19): it was already the fragile one, and it is dominated by the same few large contracts. On the 64 leaves the weighted standstill is -0.34 and -0.37 (q 0.25). Counts are a 2025 snapshot used as weights for a 2021-24 outcome and the public sector is out, so this is a check of the private-sector part.\n- **Adding the public sector (Conto annuale heads, 2021) removes the weighted association.** The 13 public leaves get 3.05 million heads (23% of the 13.2 million employees covered; the three largest of the 30 parents hold 40.7%). Weighted by employees the mean real change over all 77 groups is -5.93% (unweighted -5.81%), 86.0% of employees were in a group with a loss, and in the public leaves 99.9% (weighted mean -6.54% against -5.48% unweighted: the large public groups lost most; the police and armed-forces groups, 510 thousand heads together with the fire brigade, lost 9.5% and 9.3%). On the 30 parents the weighted correlation of the proxy with the real change falls to -0.45 and -0.41 for staleness (p 0.21 and 0.24, q 0.47) and -0.16 and -0.15 for standstill (q 0.68), from -0.52 to -0.58 unweighted on the same parents; with n_eff 11.0. On the 77 leaves the weighted standstill is -0.30 and -0.32 (q 0.155) and staleness about -0.12. The renewal-date family is not weighted-significant either (q 0.86). So the association of the equal-weight tests, and the private-sector weighted result above (q 0.048 for staleness), are not robust to giving the public sector its employee weight: the largest public groups (schools -7.6%, health service -3.6%, regions and local bodies -5.5%, ministries -5.5%) have old agreements but moderate losses, and a parent's real change is the unweighted mean of very different leaves (the Presidency of the Council gained 5.6%). The shuffled-outcome null of the weighted family is calibrated (0.03 discoveries per shuffle, 2.9% of p below 0.05). Limits: heads for 2021 and 2025 mixed, the regions and the schools of Bolzano and Trento (own contracts) not in the weights, Carabinieri counted with the police, parents averaging leaves without weights.\n- **Averaging the leaves by employees inside a parent brings the staleness association back; the choice is not neutral.** In F2 a parent's outcome was the equal-weighted mean of its leaves and its weight the sum of their employees. Averaging the leaves by employees instead moves a few parents a lot (commerce and services Z1240 from -10.03% to -7.50%, the police, armed forces and fire group Z2540 from -6.86% to -8.88%) and gives, with 2021 heads, a weighted rank correlation of -0.57 (to 2024) and -0.59 (to 2025) for staleness (q 0.028; n_eff 11.0) and -0.38 and -0.40 for standstill (q 0.088). The same with equal-weighted parent means (F2) was -0.45 and -0.41 (q 0.47). These are variants chosen after F2 was seen, so they are alternative specifications of one question, not independent confirmation; the version with consistent weights is the more coherent one, and the result rests on it. **The weighting year and the extra contracts change nothing:** with 2024 heads, with the special-statute regions and the Bolzano and Trento schools added, or both, the weighted staleness correlation stays between -0.565 and -0.572 (q 0.028-0.029 with weighted parent means, 0.47-0.50 with unweighted ones) and standstill between -0.367 and -0.379 (q 0.088-0.093). The renewal-date family, weighted, stays below significance (months expired -0.39 to -0.41, q 0.29-0.32, n_eff 7.3), and on the 77 leaves the weighted standstill is -0.29 to -0.30 (q 0.155-0.157). Reading the whole of section F together: the staleness association survives employee weighting only when the weights are applied consistently inside the parents, it is carried by a handful of large agreements (n_eff 11), and the equal-weight result for agreement groups is not a result for employees as a whole.\n- **Limits.** The groups are Ateco sectors, not agreements, and the weights (sections F-F3) are headcounts, private from 2025 and public from 2021; 24-30 parent groups give little power for anything but a strong association; contractual wages exclude wage drift, bonuses outside the agreements and one-off payments; the number of workers per agreement is not in the flows; the mapping of Istat groups to CNEL contracts is mine, the metalworkers', air-transport and public contracts cover several groups each, one archive date may be a minor agreement (commerce 2019) and ten cycles lack a period; the real change uses national HICP, and the sector deflator is a consumer-price proxy. Associations only: an old agreement may stand for other sector characteristics, and section E suggests the sector is the main one.
"""))

NOTEBOOKS["40_eu_funds_nuts1"] = [
    (M, """# EU cohesion payments, and Germany and Belgium in the "why the South" family
Two loose ends of notebook 34 are closed here. **(1) EU funds by region.** The notebook asked what sets the South apart; one candidate is the support it received. The Commission's regionalised payment data (`sources/eu_payments.py`: ERDF, Cohesion Fund and ESF payments of the 2007-2013 programme mapped to NUTS2, translated to NUTS 2021) give, for the regions of every member state, the cumulative support **per inhabitant** and **as a share of regional GDP** over the programme years. OpenCoesione (Italy's project files, with national co-financing and the 2014-2020 cycle) was not used: it needs project files of hundreds of megabytes and covers Italy only. **(2) Germany and Belgium in the EQI family.** The quality-of-government index exists only at NUTS1 for the two countries (16 Laender, 3 regions), so notebook 34 left them out; here their outcomes are aggregated from NUTS2 to NUTS1 (sums for extensive indicators, population-weighted means for rates, `analysis/nuts1.py`) and the EQI family is rerun with them.

**Design, funds.** Predictors: cohesion payments per inhabitant and as % of GDP. Outcomes: the change between the 2005-07 window mean and the 2015-19 one of log GDP per head (PPS), the employment rate, the unemployment rate, GDP per hour and hours per inhabitant. Everything is demeaned by country and purged of the initial log income (funds go to poorer regions: the question is whether, among regions of one country with the same starting income, those with more support moved differently). Spearman with the within-country permutation p, BH over the ten tests (two predictors times five outcomes), leave-one-country-out, spatial filtering of every column, Italy's slope against the other countries' (placebo ranking), a shuffled null, and variants (all funds with the EAFRD, the pre-2004 member states only, no income control, a 2005-09 baseline).

**Design, Germany and Belgium.** The EQI family of notebook 34 (the composite and its three components against the six regional gaps, four waves, plain and beyond income) without and with the two countries at NUTS1. They enter only at NUTS1 (their NUTS2 rows are dropped) and only through the EQI predictors: the structural business statistics do not exist at NUTS1 in the project, so pairs with those predictors lose the two countries and are left out of the comparison. Spatial filtering is not applied (it needs NUTS1 boundaries). Descriptive associations only: funds go where regions are poor for reasons that also drive their outcomes."""),
    (C, '''from cifresignificative.analysis import eu_funds as EFM, experiments as ex, regional_corr as rcm, regional_south as rs
fres = ex.run_funds_all()
desc = fres["f1_funds_by_region"].set_index("geo")
print("regions with cohesion payments in the 2007-2013 programme and population/GDP data:", int(desc.funds_cohesion_per_head.notna().sum()), "| countries:", desc.country.nunique())
print("median cohesion payments per inhabitant (EUR, cumulative) and % of GDP by country, top 8:"); display(desc.groupby("country")[["funds_cohesion_per_head", "funds_cohesion_pct_gdp"]].median().sort_values("funds_cohesion_per_head", ascending=False).head(8).round(1).T)
it = desc[desc.country.eq("IT")].sort_values("funds_cohesion_per_head", ascending=False)
print("Italian regions (EUR per inhabitant; % of GDP):"); display(it[["funds_cohesion_per_head", "funds_cohesion_pct_gdp"]].round(1).T)
cols = ["x", "y", "n", "rho", "p", "q_value", "loo_min_abs_rho", "loo_sign_flips"]
print("PRIMARY family (within country, beyond initial income; within-country permutation p; BH over the ten tests):"); display(fres["f2_family"][cols].round(3))
fl = fres["f3_family_filtered"]
print("after spatial filtering of every column:"); display(fl[["x", "y", "n", "rho", "p", "q_value", "rho_filtered", "p_filtered", "q_value_filtered", "same_sign"]].round(3))
print("variants:"); display(fres["f4_variants"][["variant", "x", "y", "n", "rho", "p", "q_value"]].round(3))
print("Italy's slope against the other countries' (p_rank = share of countries at least as far from the others; plain and filtered):"); display(fres["f5_italy_rank"][["x", "y", "n_countries", "slope_diff", "p_rank", "p_rank_filtered"]].round(3))
nl = fres["f6_null"]; print("shuffled null,", len(nl), "shuffles: mean discoveries", round(float(nl.discoveries.mean()), 3), "| shuffles with a discovery", round(float((nl.discoveries > 0).mean()), 3), "| share of p < 0.05", round(float(nl.share_p_below_005.mean()), 3))
fam = fres["f2_family"]; print("discoveries:", int(fam.significant.sum()), "of", len(fam), "| smallest q:", round(float(fam.q_value.min()), 3), "| rho range:", round(float(fam.rho.min()), 3), "to", round(float(fam.rho.max()), 3))
from cifresignificative.sources import eu_payments as epm
reg21 = pd.read_parquet(PROCESSED / "regional_nuts2021.parquet"); Wf = EFM.build_table(reg21, epm.load_period())
fig, axes = plt.subplots(1, 2, figsize=(11, 4)); isit = Wf.index.str[:2] == "IT"
for ax, y, lab in ((axes[0], "employment_rate_20_64", "change in employment rate, pp"), (axes[1], "unemployment_rate", "change in unemployment rate, pp")):
    ax.scatter(Wf.loc[~isit, "funds_cohesion_per_head"], Wf.loc[~isit, y], s=12, color="#2a78d6", alpha=.5); ax.scatter(Wf.loc[isit, "funds_cohesion_per_head"], Wf.loc[isit, y], s=26, color="#d94f3d", label="Italy")
    ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5); ax.set_xlabel("cohesion payments per inhabitant, deviation from the country mean (EUR)"); ax.set_ylabel(lab + ", beyond initial income")
axes[0].legend(fontsize=7); fig.suptitle("EU cohesion payments 2007-13 and the change of regional outcomes 2005-07 to 2015-19 (within countries)", fontsize=9); plt.tight_layout(); plt.show()'''),
    (C, '''nres = ex.run_nuts1_south()
n1c, n2c, n3c = nres["n1_family_comparison"], nres["n2_italy_rank"], nres["n3_de_be_within"]
e = n1c[n1c.x.str.startswith("eqi")]
g = e.groupby(["wave", "income_control"]).agg(tests=("rho_baseline", "size"), survivors_baseline=("significant_baseline", "sum"), survivors_with_de_be=("significant_with_de_be", "sum"), regions_baseline=("n_baseline", "median"), regions_with_de_be=("n_with_de_be", "median"))
print("EQI pairs (composite and three components against the six gaps), baseline against Germany and Belgium added at NUTS1:"); display(g)
print("sign changes of rho:", int((np.sign(e.rho_baseline) != np.sign(e.rho_with_de_be)).sum()), "of", len(e), "| largest change in rho:", round(float((e.rho_baseline - e.rho_with_de_be).abs().max()), 3), "| survivors lost:", int((e.significant_baseline & ~e.significant_with_de_be.fillna(False)).sum()), "| gained:", int((~e.significant_baseline.fillna(False) & e.significant_with_de_be.fillna(False)).sum()))
pr = e[(e.wave == rs.PRIMARY_WAVE) & ~e.income_control & e.x.eq("eqi")][["y", "n_baseline", "rho_baseline", "q_value_baseline", "n_with_de_be", "rho_with_de_be", "q_value_with_de_be"]]
print("2017 wave, composite EQI, plain:"); display(pr.round(3))
ip = n2c[n2c.x.str.startswith("eqi")].pivot_table(index=["x", "y"], columns="geography", values="p_diff")
print("Italy's rank among the countries on the EQI pairs: smallest p_rank baseline", round(float(ip.iloc[:, 0].min()), 3), "| with DE and BE", round(float(ip.iloc[:, 1].min()), 3), "| countries on the EQI pairs", int(n2c[n2c.geography.eq("baseline") & n2c.x.str.startswith("eqi")].n_countries.max()), "->", int(n2c[~n2c.geography.eq("baseline") & n2c.x.str.startswith("eqi")].n_countries.max()))
print("within Germany (16 Laender) and Belgium (3 regions), 2017 wave, beyond income, composite EQI (Belgium has 3 regions: shown, not interpreted):"); display(n3c[n3c.x.eq("eqi")].round(3))'''),
]

NOTEBOOKS["40_eu_funds_nuts1"].append((M, """## Takeaways (EU cohesion payments 2007-2013 and the "why the South" family with Germany and Belgium; exploratory and descriptive)
- **The South received much more support per head than the North, but less than the Central and Eastern member states.** Median cumulative cohesion payments of the 2007-2013 programme per inhabitant: Hungary 2,720 euro, Czechia 2,583, Estonia 2,555, Portugal 2,228, Greece 2,207; Italian regions: Calabria 1,543, Sicily 1,260, Puglia 1,216, Campania 1,042, Basilicata 898, Sardinia 597, against 57 for Lombardy. 224 regions of 25 countries have payments and the population and GDP needed.
- **No link between the support and the later change of regional outcomes can be shown.** Within countries and beyond the initial income, 0 of the 10 pre-specified tests (two measures of support times five outcomes) survive false-discovery control (smallest q 0.072; rho between -0.15 and 0.20). The signs are those of a helpful effect but weak: per head of population, rho 0.11 with the change in log GDP per head, 0.12 with the employment rate and -0.15 with the unemployment rate; support as a share of GDP gives 0.02, 0.08 and -0.06. The strongest is hours worked per inhabitant (0.19 and 0.20, q 0.072). After spatial filtering of every column the income and employment links shrink to 0.05 and 0.06 for the per-head measure and hours per inhabitant stays (0.21, q 0.053). The shuffled null gives a discovery in 3% of 100 shuffles.
- **Variants.** Among the pre-2004 member states alone, support per head goes with the employment rate (0.18) and the unemployment rate (-0.18), both q 0.095, still not significant; with a 2005-09 baseline nothing comes close (q 0.105 or more); with all funds including the EAFRD the income link is 0.14 (q 0.134). Without the income control, support is negatively related to the change in GDP per hour (-0.19 and -0.28, q 0.033 and 0.001): that outcome is a change in euro, larger in absolute terms in richer regions, which receive less support, so this mostly restates the initial income and is not interpreted.
- **Italy is not unusual.** Italy's slope is never among the most extreme of the 14-15 countries (smallest p_rank 0.133 plain); after filtering the one pair near the edge is the unemployment rate against support per head (0.071), a single pair among ten.
- **Germany and Belgium change nothing in the EQI result.** With Germany (16 Laender) and Belgium (3 regions) added at NUTS1 the EQI comparisons (2017 wave, plain) rest on about 198 regions instead of about 179; the survivors are 23 of 24 in the 2017 wave in both versions (18 against 17 in 2010, 21 against 20 in 2013, 22 against 18 in 2021; beyond income 12 against 12 in 2010, 2017 and 2021 and 16 against 14 in 2013), 6 of 192 correlations change sign and the largest change in rho is 0.11; the composite EQI against the employment rate is 0.366 without them and 0.373 with them (2017). Italy's smallest rank among the countries on the EQI pairs is 0.43 and then 0.40. Within Germany itself (16 Laender, beyond income) the EQI goes with a lower unemployment rate (-0.41) but not with income or employment (-0.01 and 0.09); Belgium has three regions, too few to read.
- **Limits.** Payments are the Commission's, by calendar year and by NUTS 2013 region mapped to NUTS 2021 (regions that were split, such as Zagreb, Budapest and Warsaw, are left out), not the spending on the ground, and Interreg is not included; funds go to regions for reasons that also drive their outcomes, so the income control cannot make this a causal test; OpenCoesione (national co-financing, 2014-2020) was not used. The NUTS1 outcomes of Germany and Belgium are aggregated from NUTS2, with population weights for rates (an approximation), the structural business statistics do not exist at NUTS1 here so pairs with those predictors are not compared, spatial filtering is not applied to the extension, and the EQI at NUTS1 is a coarser measure than at NUTS2.
"""))

NOTEBOOKS["41_budget_placebo"] = [
    (M, """# Is Italy's deficit surprise unusual? A placebo ranking among the EU27
Notebook 38 looked at Italy alone: how the budget laws were financed and how the deficit compared with the government's own targets. With ten years and one country it cannot say whether the misses were large *for Italy* or typical of the period, and a single country has no clustered test (Italy is one cluster). The project's rule for such claims is a **placebo ranking**: compute the same quantity for every other member and see where Italy falls.

**Reference.** The programmatic targets of notebook 38 exist only for Italy (draft budgetary plans differ in layout and content), so here the reference is the **European Commission's autumn forecast** of the year before the budget year (`sources/ec_forecasts.py`: the statistical annexes of autumn 2016 to 2025, one table per variable for all member states, so budget years 2017-2025; the autumn 2015 annex, in the old layout, adds budget year 2016 as a sensitivity window only). It is published after the draft budgetary plans and normally includes their measures: an outside view of the plan, not the government's own target. For Italy the two references are put side by side below.

**Surprise** = outturn (Eurostat, latest vintage) minus forecast, in % of GDP, for net borrowing and for its parts (revenue shortfall, primary-expenditure overrun, interest overrun); positive = worse than forecast. Two measures: the raw surprise and the **idiosyncratic** one, minus the median of the *other* 26 members in the same year (the common shocks of 2020-23 move everyone). **Ranks** are among the 27 members (1 = largest overrun, one-sided p = rank / 27, so the smallest possible p is 0.037). Windows fixed before the ranks were computed: 2017-25 without 2020, and 2021-23 (the years of the building credits); BH over the 16 tests of these two windows (4 components, 2 measures); the other windows are shown without a q. A per-country sign-flip screen of the mean idiosyncratic surprise, with a sign-randomised null, checks the machinery.

**Limits to keep in mind.** The Eurostat outturn is the latest vintage: revisions (for Italy the treatment of the building credits, decided in 2023-24) enter the surprise, so part of it is accounting, not policy. The ratios are in % of GDP and a GDP error moves them; the Italian decomposition of notebook 38 separates that term, here a denominator split at the end of the robustness section does it for every member (forecast GDP built from the forecast growth rates, see there). Cyprus 2018 (bank support) and similar one-offs are in the sample. Everything is descriptive: a forecast error is not a policy effect."""),
    (C, '''from cifresignificative.analysis import budget_xc as BX, budget as BU, experiments as ex
xres = ex.run_budget_xc_all()
F = BX.load_forecasts(); print("forecast table:", len(F), "rows; vintages", sorted(F.vintage.unique()), "| variables", sorted(F.variable.unique()))
x1 = xres["x1_surprises"]; S = x1[x1.component.eq("deficit")].pivot(index="year", columns="iso3", values="value")
print("countries with a deficit surprise in every year 2017-25:", int(S.loc[2017:].notna().all().sum()), "of", S.shape[1], "| in 2016 (autumn 2015 annex):", int(S.loc[2016].notna().sum()))
print("deficit surprise by year (pp of GDP, positive = larger deficit than forecast): Italy against the other 26"); 
tab = pd.DataFrame({"Italy": S.ITA, "median_others": S.drop(columns="ITA").median(axis=1), "min_others": S.drop(columns="ITA").min(axis=1), "max_others": S.drop(columns="ITA").max(axis=1)}); display(tab.round(2))
print("accounting identity, |deficit - (revenue + primary expenditure + interest parts)| over all country-years:", xres["x9_identity"].round(3).to_dict("records")[0])
yr = xres["x3_yearly_ranks"]; print("Italy's rank by year, deficit surprise (of 27):", yr[(yr.component == "deficit") & (yr.measure == "raw")].set_index("year")["rank"].to_dict())
fig, ax = plt.subplots(figsize=(9, 4)); o = S.drop(columns="ITA")
ax.boxplot([o.loc[y].dropna() for y in S.index], positions=S.index, widths=.5, showfliers=True, flierprops=dict(markersize=3)); ax.plot(S.index, S.ITA, "o-", color="#d94f3d", label="Italy"); ax.axhline(0, color="k", lw=.5)
ax.set_ylabel("outturn minus Commission forecast, pp of GDP"); ax.set_title("Deficit surprise against the autumn forecast of the year before, other 26 members (boxes) and Italy", fontsize=9); ax.legend(fontsize=8); plt.tight_layout(); plt.show()'''),
    (M, """## Italy's rank, by component and window"""),
    (C, '''r = xres["x2_italy_rank"]
print("Italy's rank among the 27 (1 = largest overrun); primary windows carry a BH q over the 16 tests:"); display(r[r.measure.eq("raw") | r.measure.eq("idiosyncratic")].set_index(["component", "measure", "window"])[["value", "median_others", "rank", "n", "p", "q_value"]].round(3))
prim = r[r.primary]; print("primary family:", len(prim), "tests | smallest p", round(float(prim.p.min()), 3), "| smallest q", round(float(prim.q_value.min()), 3), "| tests with q <= 0.05:", int((prim.q_value <= 0.05).sum()))
print("rank 1 or 2 among 27:"); display(r[r["rank"] <= 2][["component", "measure", "window", "rank", "p", "q_value"]].round(3))
print("sensitivity, adding budget year 2016 (autumn 2015 annex; not primary, no q): deficit, Italy's rank among 27"); display(r[r.window.eq("2016-25 without 2020") & r.component.eq("deficit")][["measure", "value", "median_others", "rank", "p"]].round(3))
print("Italy's deficit surprise in 2016 (raw):", round(float(S.loc[2016, "ITA"]), 2), "| median of the other 26:", round(float(S.loc[2016].drop("ITA").median()), 2), "| rank by year:", yr[(yr.component == "deficit") & (yr.measure == "raw") & (yr.year == 2016)]["rank"].tolist())'''),
    (M, """## Robustness: euro-area members only, other horizons, and the two references for Italy"""),
    (C, '''ea = xres["x4_euro_area_rank"]; print("deficit surprise, Italy's rank among the 20 euro-area members (p only: a q over four tests would mislead):"); display(ea.set_index(["measure", "window"])[["value", "median_others", "rank", "n", "p"]].round(3))
print("horizon (same years 2018-25 for every lead; lead 0 = the autumn forecast of the budget year itself, 2 = two autumns before):"); display(xres["x5_lead"].set_index(["lead", "window", "measure"])[["value", "median_others", "rank", "n", "p"]].round(3))
tv = xres["x6_two_views"].set_index("year"); print("Italy: miss against the Commission's forecast and against the government's own target (pp of GDP):"); display(tv.round(2))
print("mean gap between the two references (Commission deficit forecast minus government target), 2017-25:", round(float(tv.reference_gap.mean()), 2), "| 2021-23:", round(float(tv.loc[2021:2023, "reference_gap"].mean()), 2))'''),
    (C, '''loo = [{"years": "-".join(map(str, w)), "measure": m, **BX.italy_rank(X, w)} for w in ([2021, 2022], [2022, 2023], [2021, 2023]) for m, X in (("raw", S), ("idiosyncratic", BX.idiosyncratic(S)))]
print("the 2021-23 window with one year left out (deficit surprise, Italy's rank among 27):"); display(pd.DataFrame(loo).set_index(["years", "measure"])[["value", "median_others", "rank", "n", "p"]].round(3))
print("Italy's rank in the single years 2022 and 2023 (raw):", {int(y): int(yr[(yr.component == "deficit") & (yr.measure == "raw") & (yr.year == y)]["rank"].iloc[0]) for y in (2022, 2023)})'''),
    (M, """## Robustness: an early outturn instead of the latest vintage"""),
    (C, '''fe, rv = xres["x10_first_estimate"], xres["x10_revisions"]
print("deficit surprise with the Commission's estimate of the year just ended (autumn forecast of B+1) or the latest Eurostat vintage as the outturn, budget years 2017-24; Italy's rank among 27:"); display(fe.set_index(["outturn", "window", "measure"])[["value", "median_others", "rank", "n", "p"]].round(3))
print("revision of the deficit after the early estimate (latest minus early, pp of GDP; positive = larger deficit), Italy against the others:"); display(rv.set_index("year").round(2))'''),
    (M, """## Robustness: the GDP denominator
The surprise is a difference of ratios, so a GDP forecast error moves it. Forecast nominal GDP of the budget year B is rebuilt from the latest outturn of B-2 and the Commission's autumn forecast of B-1 for real growth and the GDP deflator in B-1 and B (the annexes give growth rates, not levels). With D the deficit in euro, surprise = (D_out - D_fc) / G_out + (D_fc / G_fc) (G_fc / G_out - 1): the second term (`denominator`) is the part due to the GDP error alone, the first (`level`) is the miss of the deficit in euro at the outturn GDP. Ranks only (primary windows), no q: this is a decomposition of the tested quantity, not a new family."""),
    (C, '''dr, sp = xres["x11_denominator_ranks"], xres["x11_italy_split"]
print("Italy's rank among 27 for the deficit surprise and its two parts (1 = largest):"); display(dr.set_index(["part", "measure", "window"])[["value", "median_others", "rank", "n", "p"]].round(3))
print("Italy by year (pp of GDP): surprise = denominator + level:"); display(sp.set_index("year").round(2))'''),
    (M, """## Which countries are unusual? Per-country screen and its null"""),
    (C, '''sc = xres["x7_screen"]; print("mean idiosyncratic deficit surprise, 2017-25 without 2020 (sign-flip p, BH q over countries); top 8 and bottom 3:"); display(pd.concat([sc.head(8), sc.tail(3)]).round(3))
print("discoveries (q <= 0.05):", int(sc.significant.sum()), "of", len(sc), "| Italy's rank in the screen:", int(sc.index[sc.iso3.eq("ITA")][0]) + 1)
nl = xres["x8_null"]; print("sign-randomised null,", len(nl), "shuffles: mean discoveries", round(float(nl.discoveries.mean()), 3), "| shuffles with a discovery", round(float((nl.discoveries > 0).mean()), 3), "| share of p < 0.05", round(float(nl.share_p_below_005.mean()), 3))
r2 = xres["x2_italy_rank"]; comp = r2[(r2.measure == "idiosyncratic") & (r2.window == "2021-23")].set_index("component")[["value", "rank", "p"]]
S3 = {c: x1[x1.component.eq(c)].pivot(index="year", columns="iso3", values="value") for c in BX.COMPONENTS}
big = pd.DataFrame({c: BX.window_mean(BX.idiosyncratic(s), [2021, 2022, 2023]) for c, s in S3.items()}).dropna()
print("2021-23, idiosyncratic means, the five largest deficit overruns and their parts:"); display(big.sort_values("deficit", ascending=False).head(5).round(2))'''),
]

NOTEBOOKS["41_budget_placebo"].append((M, """### The members' own draft budgetary plans as a second reference
The Commission's forecast is the outside view; the government's own plan is the inside one. `sources/dbp_euro.py` downloads the English copies of the October draft budgetary plans of the euro-area members from the Commission's yearly pages (119 files, 2018-2025) and reads the planned net lending of the general government from the macro table: three B.9 rows whose values add up (rest of the world = private sector + general government) find the right row whatever its label. 51 plans are verified this way (9 countries: Austria, Cyprus, Greece, Italy, Lithuania, Latvia, Malta, the Netherlands, Portugal; 4-9 a year), ten more are label-only and left out, 58 are not parsed (new templates, fractions, wrapped labels). Italy's eight parsed plans match `deficit_targets.csv`. Three identity-only rows of 2021 that differ from the Commission's forecast of that autumn by 2.4-6.3 points are dropped as misreadings. The sample is thin, so each rank is shown next to the Commission-based rank for exactly the same country-years.
""")); NOTEBOOKS["41_budget_placebo"].append((C, '''dp = xres["x12_dbp_plans"]; dr = xres["x13_dbp_ranks"]
print("plans used (country-years):", len(dp), "| countries:", sorted(dp.iso3.unique()), "| plans per year:", dp.groupby("year").size().to_dict())
print("Italy's plan and surprise against it by year (net lending, % of GDP):"); display(dp[dp.iso3.eq("ITA")].round(2))
print("Italy's rank among the countries with plans (1 = the largest overrun), own plan against the Commission forecast on the same country-years:"); display(dr.drop(columns="countries").round(2))
'''))
NOTEBOOKS["41_budget_placebo"].append((M, """## Takeaways (deficit surprise against the Commission's autumn forecast, 27 members, budget years 2017-2025; descriptive)
- **Data.** The statistical annexes of the ten autumn forecasts of 2016-2025 give net lending, revenue, expenditure and interest for all 27 members (the autumn 2015 annex has the old layout and is read separately: it adds budget year 2016, used only as a sensitivity window, so the primary windows start in 2017). Every member has a surprise in every year, and the three parts add up to the deficit surprise within 0.05 points on average (largest gap 0.4: rounding of the published figures). As a check of the parsing, the estimate column of each annex agrees with the latest Eurostat outturn within 0.08 to 0.31 points on average.
- **Adding budget year 2016 changes nothing.** With the autumn 2015 annex as a sensitivity window (2016-25 without 2020, not primary, no q) Italy is second of 27 on the deficit surprise (raw +0.53, median of the others -0.73; p 0.074), as in the primary window; Italy's own surprise in 2016 is +0.10 against a median of -0.85 for the others (rank 5).
- **Italy's deficit overrun of 2021-23 was the largest of the 27.** Against the autumn forecast of the year before, the outturn deficit was above the forecast by 2.30 points of GDP on average (median of the other 26: -1.18); minus the other members' yearly median it is 3.53 (the next are Hungary 2.83 and Latvia 2.30). The one-sided placebo p is 1/27 = 0.037, the smallest the design allows, and it does not survive false-discovery control: over the 16 pre-specified tests (two windows, four components, two measures) the smallest q is 0.296 and none reaches 0.05. Among the 20 euro-area members Italy is also first (p 0.05).
- **The rank does not hinge on one year or one horizon.** In 2022 and 2023 taken singly Italy is first (+2.3 and +3.5 against medians of -1.25 and -0.70 for the others; 2021 is third, +1.1). Leaving one year out of 2021-23 gives rank 2 (2021-22) or 1 (2022-23, 2021 and 2023). Against the forecast of the budget year itself the rank is still first (1.43 points); against the forecast of two years before it is second (3.70).
- **The result does not depend on the outturn vintage.** With the Commission's own estimate of the year just ended (autumn forecast of B+1, budget years 2017-24) instead of the latest Eurostat figures, 2021-23 is still first of 27 (raw +1.73, median of the others -1.05; idiosyncratic +3.02) and the period without 2020 still second (raw +0.47). What the later revisions changed is mostly one number: Italy's 2021 deficit was revised up by 1.7 points of GDP (the other members' median -0.1, largest +0.2), which is the accounting treatment of the building credits; in every other year Italy's revision is within 0.2.
- **A GDP forecast error does not explain it; at constant GDP it is larger.** Rebuilding forecast nominal GDP from the forecast growth and deflator, the denominator term is negative in Italy in 2021-23 (-0.47, -0.30, -0.17: inflation made the economy larger than forecast, which lowers every ratio), so the miss of the deficit in euro at the outturn GDP is +1.57, +2.60 and +3.67 points; for 2021-23 Italy is first on that level miss (raw 2.62, idiosyncratic 3.62, p 0.037) and the denominator term itself ranks 21 of 27. Without 2020 the whole period is also first on the level miss (raw 0.71, p 0.037) where the plain surprise is second (0.59). The forecast GDP is a rebuilt level, not a published one (latest outturn of B-2 times forecast growth), so these figures carry that approximation.\n- **Outside those years Italy is unremarkable.** 2017-19 rank 10 (raw -0.30, median of the others -0.50), 2024-25 rank 15. Without 2020 the whole period gives rank 2 (0.59 raw, 1.39 idiosyncratic; p 0.074, q 0.296): the 2021-23 years carry it. In 2020 the pandemic overrun was 7.1 points, rank 8 (median of the others 5.65).
- **It is spread over the parts, not one of them.** In 2021-23 the idiosyncratic means are 1.77 for the revenue shortfall (rank 5 of 27), 2.05 for primary expenditure (rank 4) and 0.30 for interest (rank 3); none of them is first, and none has a q below 0.29. Other members have larger single parts (Latvia 4.15 and Austria 2.95 on primary expenditure; Bulgaria 2.40 on revenue), so what is unusual in Italy is the sum. This is compatible with the Italian account of notebook 38 (capital transfers for the building credits) but does not test it: no other country has that item here, and the outturn is the latest Eurostat vintage, whose treatment of the credits was revised, so part of the surprise is accounting.
- **The two references agree for Italy.** The Commission's deficit forecast was on average 0.13 points above the government's own target (0.03 in 2021-23; the largest gaps +0.8 in 2021 and -0.9 in 2023), so the miss is about the same against either (2021-23: +1.1, +2.3, +3.5 against the Commission, +1.9, +2.5, +2.6 against the government).
- **The country screen cannot discover anything by design; it only orders.** The sign-flip screen of the mean idiosyncratic surprise (2017-25 without 2020) finds 0 of 27 discoveries, but with 8 years the smallest attainable p is about 0.008 and the smallest q over 27 countries about 0.11 (reached by Denmark, -2.48, and the Netherlands, -1.03, which tie): a real persistent difference could not pass either. The order is still informative: Hungary +1.51 (p 0.016, q 0.14) first and Italy second (+1.39, p 0.112, q 0.303), with Denmark and the Netherlands at the bottom (deficits smaller than forecast). The sign-randomised null gives 0 discoveries in 200 runs and 4.7% of p-values below 0.05, so the p-values are calibrated even if the screen has no power at this size.
- **The governments' own plans give the same placement, on a thin sample.** From 51 verified draft budgetary plans of 9 euro-area members (4-9 a year), Italy's 2021-23 deficit exceeded its own plan by 2.70 points on average (4.00 against the median of the others in the same year): first of 8 countries (p 1/8; the Commission-based figure on the same country-years is 2.30, first of 8 too). Over 2018-25 without 2020 it is 0.96 (1.84 idiosyncratic), first of 7; the Commission-based rank on those country-years is first raw and second idiosyncratic. Italy's 2023 plan is the outgoing government's October one (-3.4; the November update said -4.5, which would shorten the 2023 surprise from 3.7 to 2.6). No test is possible with 7-8 countries (the smallest p is 1/8) and the countries with parsable plans are not a random draw: this is a second look, not a confirmation. The plan table: `data/external/dbp_targets.csv`.\n- **Limits.** One-sided ranks among 27 cannot go below 0.037 and Italy is one country, so this is a placement, not a calibrated test; the 16 tests are strongly dependent (BH is conservative here). The reference is the Commission's forecast (policy as known in the autumn), not each government's plan; ratios are in % of GDP and a GDP error moves them (separated in the denominator split, which rests on a forecast GDP level rebuilt from the latest outturn of B-2 and the forecast growth, not a published level); Cyprus 2018 and similar one-offs are in the sample; ten years, with 2020 and the 2021-23 recovery dominating. All associations are descriptive: a forecast error is not a policy effect.
"""))


NOTEBOOKS["42_firm_size"] = [
    (M, """# Firm size and productivity: where does Italy's size structure stand, and how much of the gap does it carry?
Notebook 35 (regulation) and notebooks 08, 13 and 23 (sectors) left one question open: *inside* a sector, is Italy's productivity shortfall about the **average firm**, about a **tail of small firms**, or about **how employment is spread over firm sizes**? The firm-level dispersion statistics that answer this best (CompNet, IWH Halle) are not available to this project (access is for researchers with an institutional affiliation). This notebook uses what is public instead: the **Eurostat structural business statistics by employment size class** (`sbs_sc_sca_r2`, 2005-2020, and `sbs_sc_ovw`, 2021 onwards): for each country, year and NACE section, the number of enterprises, persons employed and value added in five disjoint classes (0-9, 10-19, 20-49, 50-249, 250+ persons).

**What it can and cannot do.** It gives the size *mix* and the *average* productivity of each class, so it can say how much of a gap is the mix of employment over sizes and how much is productivity within a size class. It cannot see the dispersion of firms inside a class (the CompNet question), nor which firms are missing: a country with the same class averages and a different distribution inside classes looks the same here.

**Design.** (1) Italy's employment shares and productivity by class, and its **rank among the EU27 members** (placebo ranking: Italy is one cluster, so no clustered test; the smallest two-sided p with n members is 2/n, so with about 23 members nothing can reach q < 0.05 and the results are orderings, as in notebook 41). (2) The **mix effect**: the log of Italy's productivity recomputed with the other members' median size mix over its actual one, with the same quantity for every member. (3) An **exact decomposition** of Italy's log gap to a reference (each of Germany, France, Spain, the Netherlands, and the median of the others) into `size_mix` (Italy's mix against the reference mix, at Italy's own class productivities) and `within` (Italy's class productivities against the reference's, at the reference mix). (4) The same by NACE section, and a sector-versus-within-sector split of the micro-firm share.

**Limits to keep in mind.** The statistical unit is the *enterprise* as each country's register defines it (a group split into legal units looks like many small firms); the persons are heads, not hours, and the productivity is value added at factor cost (2005-2020) or basic prices (2021 onwards) per person, so the two vintages are never combined. Euro levels are not compared across countries without a correction: the level comparisons use the GDP price level index of the country-year (an approximation for value added) and, per hour, the economy-wide hours per worker (AMECO), assumed equal in every class. The `size_mix` term is the same per head and per hour (a country-wide factor cancels in it); only `within` depends on the corrections. Class productivity differences mix selection (productive firms grow) and a causal advantage of scale, so the mix effect is an *accounting* number: moving employment between classes at class-average productivity is an upper bound of what a change of structure could do, not a forecast. Descriptive only."""),
    (C, '''from cifresignificative.analysis import size_class as SC
res = SC.run_size_all(); size = SC.load_size(); S1 = res["s1_structure"]; cl = SC.CLASSES
cov = S1.groupby(["vintage", "year"]).iso3.nunique().unstack(0); print("EU27 members with all five size classes (persons and value added) in the economy aggregate:"); display(cov.T.fillna(0).astype(int))
print("Italy has the five classes in:", {v: sorted(S1[(S1.vintage == v) & (S1.iso3 == "ITA")].year.astype(int)) for v in ("old", "new")})
raw = size[(size.iso3 == "ITA") & (size.vintage == "old") & (size.nace_r2 == SC.ECONOMY["old"]) & (size.year.isin([2015, 2016, 2017, 2019, 2020]))]
print("Italy, Eurostat flags of the size-class series by year (b = break in series):", raw.groupby("year").flag.agg(lambda f: sorted(set(f.dropna()))).to_dict())
print("EU27 members without the five classes in 2019 (old vintage):", sorted(set(EU27_ISO3) - set(S1[(S1.vintage == "old") & (S1.year == 2019)].iso3)))
print("members with a break flag in the 2019 economy aggregate:", int(S1[(S1.vintage == "old") & (S1.year == 2019)].broken.sum()))'''),
    (M, """## Italy's size structure in 2019 (the main snapshot)
2019 is the last year before the pandemic with the most members (23 of 27); 2020 is reported below. Each share and relative productivity is a ratio inside the country, so no price or currency correction enters."""),
    (C, '''t = SC.class_table(size, "old"); t = t[t.year == 2019]
P = t.pivot(index="iso3", columns="size_emp", values="persons")[cl]; V = t.pivot(index="iso3", columns="size_emp", values="va_meur")[cl]; E = t.pivot(index="iso3", columns="size_emp", values="enterprises")[cl]
Sh = P.div(P.sum(axis=1), axis=0); rel = (V / P).div(V.sum(axis=1) / P.sum(axis=1), axis=0)
tab = pd.DataFrame({"employment share, Italy": Sh.loc["ITA"], "median others": Sh.drop("ITA").median(), "min others": Sh.drop("ITA").min(), "max others": Sh.drop("ITA").max(),
                    "productivity / own total, Italy": rel.loc["ITA"], "median others ": rel.drop("ITA").median(), "enterprises, Italy": E.loc["ITA"], "enterprise share, Italy": E.loc["ITA"] / E.loc["ITA"].sum()})
display(tab.round(3)); print("members:", len(Sh), "| Italy persons employed in the five classes:", int(P.loc["ITA"].sum()), "| mean persons per enterprise, Italy:", round(float(P.loc["ITA"].sum() / E.loc["ITA"].sum()), 2))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); x = np.arange(5); o = Sh.drop("ITA")
ax[0].bar(x - .2, Sh.loc["ITA"] * 100, .4, color=ITALY_C, label="Italy"); ax[0].bar(x + .2, o.median() * 100, .4, color=EU_C, label="median of the others"); ax[0].errorbar(x + .2, o.median() * 100, yerr=[(o.median() - o.min()) * 100, (o.max() - o.median()) * 100], fmt="none", ecolor="k", lw=.8)
ax[0].set_xticks(x, cl); ax[0].set_ylabel("% of persons employed"); ax[0].set_title("Employment by size class, 2019 (lines: min-max of the others)", fontsize=9); ax[0].legend(fontsize=8)
r_o = rel.drop("ITA"); ax[1].bar(x - .2, rel.loc["ITA"], .4, color=ITALY_C); ax[1].bar(x + .2, r_o.median(), .4, color=EU_C); ax[1].errorbar(x + .2, r_o.median(), yerr=[r_o.median() - r_o.min(), r_o.max() - r_o.median()], fmt="none", ecolor="k", lw=.8)
ax[1].axhline(1, color="k", lw=.5); ax[1].set_xticks(x, cl); ax[1].set_title("Value added per person / own average, 2019", fontsize=9); plt.tight_layout(); plt.show()'''),
    (M, """## Where Italy ranks, and the primary family
The statistics are the micro (0-9) and large (250+) employment shares, the productivity of those two classes relative to the country's own average, the large-to-micro **gradient** log(productivity of 250+ / 0-9), the **mix effect**, and the mean persons per enterprise. `rank` 1 = the highest value among the members with data. The primary family is these seven statistics in 2019; BH q is computed over it, other years carry no q."""),
    (C, '''pos = res["s2_position"]; main = pos[pos.primary].set_index("statistic")
print("2019, old vintage, economy aggregate (B-N without K, repairs included): Italy against the others; BH q over the", len(main), "statistics:"); display(main[["n", "italy", "median_others", "rank", "p", "q_value"]].round(3))
print("smallest attainable two-sided p with", int(main.n.min()), "members:", round(2 / main.n.min(), 3), "| smallest q in the family:", round(float(main.q_value.min()), 3), "| statistics with q <= 0.05:", int((main.q_value <= 0.05).sum()))
keep = ["micro_share", "large_share", "gradient", "mix_effect"]; o = pos[~pos.primary & pos.statistic.isin(keep)]
print("rank of Italy in the other years (not primary, no q; the old and new vintages are different series: compare ranks within a vintage only):"); display(pd.concat([pos[pos.primary & pos.statistic.isin(keep)], o]).pivot_table(index=["vintage", "year"], columns="statistic", values="rank")[keep])
print("members with data per year, same table:"); display(pos[pos.statistic == "micro_share"].set_index(["vintage", "year"])[["n", "n_break", "italy_break"]])
print("Italy's micro share of employment by year (old vintage has Eurostat break flag in 2017: 2013-2016 and 2019 are not one series; new vintage from 2021):", pos[pos.statistic == "micro_share"].set_index(["vintage", "year"]).italy.round(3).to_dict())'''),
    (M, """## The mix effect: what Italy's productivity would be with the other members' size mix
For each member, the log of its productivity recomputed with the median employment shares of the *other* members (renormalised) and its own class productivities, over its actual productivity: positive = the member's mix costs it. The calculation uses only the member's own class productivities, so prices, currency and hours do not enter."""),
    (C, '''S19 = Sh.copy(); LP19 = (V / P * 1000).where((P > 0) & (V > 0)); typ = SC.others_mix(S19).loc["ITA"]; own = S19.loc["ITA"]; w = LP19.loc["ITA"]; base = float((own * w).sum())
contrib = pd.DataFrame({"Italy share": own, "typical share (median of the others)": typ, "difference, pp": (typ - own) * 100, "Italy productivity, kEUR/person": w, "contribution to the change of productivity": (typ - own) * w / base})
display(contrib.round(3)); print("sum of the contributions:", round(float(contrib.iloc[:, -1].sum()), 4), "= exp(mix effect) - 1; mix effect (log):", round(float(np.log(1 + contrib.iloc[:, -1].sum())), 4))
mx = S1[(S1.vintage == "old") & (S1.year == 2019)].set_index("iso3").mix_effect.dropna().sort_values()
print("mix effect, 2019, all members (log points; positive = the mix costs the country):"); display(mx.round(3).to_frame("mix_effect").T)
loo = res["s9_mix_leave_one_out"]; print("Italy's mix effect leaving each other member out of the reference group:", "min", round(float(loo.mix_effect.min()), 3), "max", round(float(loo.mix_effect.max()), 3), "(", len(loo), "members )")
print("with the mean instead of the median of the others:", res["s6_mean_mix"][["italy", "rank", "n"]].round(3).to_dict("records")[0])
gd = res["s8_gap_decomposition"]; g0 = gd[(gd.section == "economy") & (gd.basis == "hour")].set_index("reference")
print("Italy's mix effect against the mix of single countries (-size_mix; the same per head and per hour):", (-g0.size_mix).round(3).to_dict())
fig, ax = plt.subplots(figsize=(9, 3.6)); cols = [ITALY_C if c == "ITA" else PEER_C for c in mx.index]; ax.bar(mx.index, mx.values * 100, color=cols); ax.axhline(0, color="k", lw=.5); ax.tick_params(axis="x", labelrotation=90, labelsize=7)
ax.set_ylabel("log points x 100"); ax.set_title("Mix effect, 2019: productivity gain with the median size mix of the other members (Italy in blue)", fontsize=9); plt.tight_layout(); plt.show()'''),
    (M, """## Levels and the exact gap decomposition
Levels are compared only after two corrections that are approximations: the GDP price level index of the country-year (EU27 = 100) and, for the per-hour basis, the economy-wide hours per worker (AMECO). Heads, not hours, are what the survey counts, so part-time work lowers a country's value added per head; the per-hour basis undoes this on average but not class by class. The decomposition is exact whatever the corrections: total = `size_mix` + `within`."""),
    (C, '''lg = res["s7_level_gap"]; e = lg[lg.section == "economy"].set_index(["basis", "size_emp"])
print("value added per person (head) and per hour, EUR at the EU27 price level, 2019, economy aggregate; ratio of Italy to the median of the others and to four peers:"); display(e[["n", "italy", "median_others", "ratio_to_median", "rank", "ratio_DEU", "ratio_FRA", "ratio_ESP", "ratio_NLD"]].round(2))
for sec, name in (("economy", "economy aggregate"), ("C", "manufacturing (C)"), ("G", "trade (G)")):
    g = gd[gd.section == sec].set_index(["basis", "reference"])[["n", "total", "size_mix", "within"]]; print("exact decomposition of Italy's log gap,", name, "(log points; total = size_mix + within; negative = Italy below the reference):"); display(g.round(3))
print("largest departure from the identity over all rows:", float((gd.total - gd.size_mix - gd.within).abs().max()))
f = gd[(gd.section == "economy") & (gd.basis == "hour") & gd.reference.isin(SC.PEERS)].set_index("reference"); fig, ax = plt.subplots(figsize=(7, 3.4)); xx = np.arange(len(f))
ax.bar(xx - .2, f.size_mix * 100, .4, color=ITALY_C, label="size mix"); ax.bar(xx + .2, f.within * 100, .4, color=EU_C, label="within size class"); ax.plot(xx, f.total * 100, "k_", ms=18, label="total"); ax.axhline(0, color="k", lw=.5)
ax.set_xticks(xx, f.index); ax.set_ylabel("log points x 100, Italy minus reference"); ax.set_title("Italy's gap per hour to four peers, 2019", fontsize=9); ax.legend(fontsize=8); plt.tight_layout(); plt.show()'''),
    (M, """## Which sectors? Positions by NACE section and the micro-firm share split into sector mix and within-sector"""),
    (C, '''sp = res["s3_sector_position"]; keep = ["micro_share", "large_share", "gradient", "mix_effect"]
tabs = sp[sp.statistic.isin(keep)].pivot_table(index=["nace_r2", "label"], columns="statistic", values="rank")[keep]; ns = sp[sp.statistic == "micro_share"].set_index(["nace_r2", "label"]).n
ital = sp[sp.statistic == "micro_share"].set_index(["nace_r2", "label"])[["italy", "median_others"]].rename(columns={"italy": "micro share Italy", "median_others": "micro share median others"})
print("Italy's rank (1 = highest) among the members with data, by NACE section, 2019 (the five classes needed in that section):"); display(pd.concat([ns.rename("n"), ital, tabs], axis=1).round(3))
mixs = sp[sp.statistic == "mix_effect"].set_index("nace_r2")[["label", "n", "italy", "median_others", "rank", "p"]]; print("mix effect by section (log points):"); display(mixs.round(3))
comp = res["s4_composition"].set_index("iso3"); cr = res["s5_composition_rank"].set_index("statistic")
print("micro share of employment over the 9 sections C, F, G, H, I, J, L, M, N, split into sector composition and within-sector part (members with the total and 0-9 persons in every section):", len(comp)); display(cr.round(3))
print("identity check, largest |composition + within - total gap|:", float((comp.composition + comp.within - comp.total_gap).abs().max()))
display(comp.loc[["ITA", "DEU", "FRA", "ESP", "GRC", "SVK", "PRT"]].round(3))'''),
    (M, """## Robustness: the pandemic year, the new vintage, and Italy's 2017 break
The table of ranks above gives 2013-2020 (old vintage) and 2021-2024 (new vintage). Italy's 2017 values carry the Eurostat break flag, so the values for 2013-2016 and for 2019 are on different bases and are not read as a trend; the ranks are compared year by year. The new vintage has a wider scope for the economy aggregate (industry, construction and market services B-S without O and S94) and value added at basic prices, so its levels are not comparable with the old ones either."""),
    (C, '''pp = res["s2_position"]; sub = pp[pp.statistic.isin(["micro_share", "mix_effect"])].pivot_table(index=["vintage", "year"], columns="statistic", values=["italy", "median_others", "rank", "n"])
display(sub.round(3)); print("Italy's mix effect: old vintage", pp[(pp.statistic == "mix_effect") & (pp.vintage == "old")].set_index("year").italy.round(3).to_dict(), "| new vintage", pp[(pp.statistic == "mix_effect") & (pp.vintage == "new")].set_index("year").italy.round(3).to_dict())'''),
]


NOTEBOOKS["42_firm_size"].append((M, """## Takeaways (Eurostat size-class statistics, 2019 main snapshot, 23 members; descriptive)
- **Data and coverage.** The economy aggregate has the five classes for 13-23 members a year in 2010-2020 (old vintage) and 25-27 in 2021-2024 (new vintage); in 2019 Austria, Cyprus, Denmark and Portugal are missing, so the ranks are among 23. Italy has complete classes in 2013, 2015, 2016, 2019, 2020 and 2021-24; its 2017 values carry the Eurostat break flag, so 2013-2016 and 2019 are not one series and ranks are compared year by year. The five classes add up to the published total within 1%.
- **Italy has the second most micro-firm-heavy structure after Greece and Slovakia.** In 2019 41.9% of persons employed work in enterprises with fewer than 10 persons against a median of 30.4% in the other members (rank 3 of 23; the others range 17.1-44.1%), and 23.9% in enterprises with 250 or more against 31.5% (rank 19 of 23, the fifth lowest); the mean enterprise has 4.2 persons against 5.3 (rank 20). 94.4% of Italy's enterprises are micro. The productivity gradient is steep: value added per person is 0.61 of the Italian average in micro firms and 1.51 in the largest, against 0.70 and 1.29 for the median of the others, so log(250+ over 0-9) is 0.92 against 0.63 (rank 5).
- **Nothing here can pass a test, by design; these are orderings.** With 23 members the smallest two-sided placebo p is 2/23 = 0.087; over the seven statistics of the primary family the smallest q is 0.435 and none reaches 0.05 (Italy is second on the mix effect, p 0.174, and third on the micro share, p 0.261).
- **The size mix costs Italy about 10 log points, and the ranking is stable.** With the other members' median size mix and Italy's own class productivities, Italy's productivity would be 10.0 log points higher (10.5%), second after Greece (23.8) and above Slovakia (8.9); most members are within +-6 (France and Germany -6). The gain comes from moving 11.8 points of employment out of micro firms (and 1.9 out of the 10-19 class) into the 20-49 (+1.3), 50-249 (+5.2) and 250+ (+7.3) classes; leaving each other member out of the reference leaves it between 9.8 and 10.2, the mean instead of the median gives 10.3 (rank 2). In 2020 it is 10.1 (rank 2); in the new vintage 9.2 (2021), 8.5, 7.2, 7.1 (2024), rank 3-4. The 2013-2016 values (13.6, 12.5, 12.1, rank 1) are on the other side of Italy's 2017 break. This is an accounting upper bound: moving workers between classes at class-average productivity ignores that firms are larger because they are more productive.
- **The mix is a large part of the gap to Germany, France and the Netherlands, whatever the price and hours corrections.** Against a named country the size-mix term is the same per head and per hour (a country-wide factor cancels): Italy's own productivities with the German mix would be 18.2 log points higher, with the French 18.7, the Dutch 12.9, the Spanish 6.3 (about 10 with the median of the others: 10.0 per hour and 10.0 per head, on 23 members). Italy's productivity per hour (price-level adjusted, economy aggregate, 2019) is 27.6 log points below Germany's, 28.3 below France's and 26.9 below the Netherlands' (exact split: size mix -18.2, -18.7, -12.9; within size class -9.4, -9.6, -14.0). Per head the totals are -7.3, -17.2, -12.6 and the within terms +10.9, +1.4, +0.3: **the sign of the within-class term against Germany and France depends on the hours correction** (Italian workers work more hours), so the data do not settle it; only the size-mix term is robust. Italy is above Spain (+8.5 per hour, +8.7 per head), where the Spanish mix is also micro-heavy (the mix costs Italy 6.3 against it).
- **Where within-class differences show, they are in the smallest firms.** Per hour, Italy's value added per person is 0.65 of Germany's and 0.60 of France's in firms with fewer than 10 persons (0.80 and 0.67 per head), but 0.94-1.04 of Germany's and 0.85-1.04 of France's in the classes 10-249 and 0.91 and 0.97 in the 250+ class (per head, across the classes of 10 persons and more: 1.11-1.27 against Germany and 0.95-1.16 against France). Against the median of the others Italy's micro firms are at 0.97-0.98 and its larger classes at 1.16-1.56 (rank 3-9 of 23). On these approximations the data do not show a shortfall of Italy's large firms against Germany or France; they cannot exclude one of the size of the corrections (about +-10 log points).
- **The micro-firm excess is not a sector-mix artefact.** In 2019 Italy has the highest micro share of employment among the members with data in trade (55.5% against a median of 34.7%), accommodation and food (56.3%, 38.4%), professional services (73.5%, 52.5%) and real estate (93.1%, 54.3%; only 14 members), the second in construction (62.3%, 46.3%) and the third in manufacturing (21.2%, 12.5%); transport and storage (rank 17), information and communication (10) and administrative services (12) are unremarkable. Over the nine sections C, F, G, H, I, J, L, M and N the micro share is 42.4%, 10.3 points above the typical reference, of which the sector composition is -0.8 (rank 16 of 24) and the within-sector part +11.1 (rank 3 after Slovakia 14.5 and Greece 12.2); the split is an identity (largest error 0). The sector mix effects are 17.0 log points in trade (rank 2), 15.1 in accommodation and food (rank 1), 14.1 in manufacturing (rank 4), 11.4 in construction (rank 4), 11.2 in professional services (rank 3) and not positive in transport and storage (-2.9), information and communication (-0.1) and administrative services (+0.5).
- **Limits.** The statistical unit is the enterprise as the national register defines it (groups split into legal units look like many micro firms), the persons are heads not hours, the price level is GDP-wide, the class hours are the economy-wide average, and classes are averages: this notebook sees the size mix and class means, not the dispersion inside classes, so **the firm-level dispersion question that CompNet would have answered stays open**. Size and productivity are jointly determined; nothing here says that a different size structure would have raised productivity. Findings go to `reports/findings.md`, limits to `docs/analysis_framework.md`.
"""))
