"""Cells of notebook 04_demography_pensions."""
from . import NOTEBOOKS, QUOTED, C, M

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

NOTEBOOKS["04_demography_pensions"].append((M, """## Takeaways
- **Italy is the oldest EU27 country:** old-age dependency 38.4 (1st of 27), median age 48.7 (1st), fertility 1.2 (22nd).
- **Spending is above what ageing alone predicts.** Public old-age spending was 13.7% of GDP in 2022-24, 3 pp above the 10.8% implied by the other members' relationship with the dependency ratio (3rd highest in 2024). The association with older-worker employment is negative: Italy combines high spending with low 55-64 employment (59.0% vs 67.8% median in 2024).
- **But Italy has moved a lot:** 55-64 employment rose from 31.4% (2005) to 59.0%, and the effective exit age for men from 59.9 to 64.0 (EU median 63.6).
- **Design in 2024 (OECD):** net replacement rate for an average earner 79% (8th of 30; median 72%), gross 70.6% (8th; median 54%); normal retirement age 64.8 now but 70 for future entrants (rank 4-5 of 30). The retirement-age path is the main reform lever already legislated.
- **Old-age poverty is not unusual:** at-risk-of-poverty among 65+ was 17.6% against a 16.6% median in 2024 (gap about 0 sd). The pension system protects the old comparatively well; the cost falls on public finances and on younger cohorts (see notebook 05)."""))
NOTEBOOKS["04_demography_pensions"].insert(-1, (C, QUOTED + '''quoted("old_age_dependency", [2024]); quoted("median_age", [2024]); quoted("fertility_rate", [2024]); quoted("pension_spend_pct_gdp", [2024])
quoted("employment_rate_55_64", [2005, 2024]); quoted("oecd_exit_age_male", [2005, 2024]); quoted("arop_65plus", [2024])'''))
