"""Cells of notebook 26_high_frequency_dynamics."""
from . import NOTEBOOKS, C, M

NOTEBOOKS["26_high_frequency_dynamics"] = [
    (M, """# High-frequency dynamics (monthly and quarterly Eurostat series, 2005-2026)
The annual panel cannot answer timing questions. Here: 14 monthly and 4 quarterly series (unemployment, HICP, industrial production, retail, 10-year yields, sentiment, GDP growth, employment, hours, labour cost), 31 countries and aggregates.
Safeguards: every series is **transformed to be approximately stationary** before any comparison (levels of yields, indices and prices trend and would correlate by accident); cross-correlations are **prewhitened**; panel tests control for the dependent series' own past with **country and month effects and country-clustered errors**; events are judged against a **placebo distribution from every other month of the same series** and corrected for multiple testing (BH-FDR).
**Predictive order and association only, never causal.** Code: `src/cifresignificative/studies/italy_productivity/analysis/hf.py`, `experiments.py`."""),
    (C, '''from cifresignificative.studies.italy_productivity.analysis import hf as H, experiments as ex
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
- **Data.** 18 series (14 monthly, 4 quarterly) for 23-31 countries and aggregates, 2005 to September 2026. HICP comes from the ECOICOP-2 dataset (`prc_hicp_minr`) because the old `prc_hicp_manr` stops in December 2025. Levels are mostly non-stationary (only 10% of countries pass ADF+KPSS for the inflation rate and 21% for labour-cost growth), while the transformed series pass in 68-100% of countries (sentiment levels are the weakest, 68-78%), so all tests use changes or growth rates.
- **The Italy-Germany spread** peaked at 5.19 pp in November 2011 (Monti's government took office that month) and stood at 0.89 pp in September 2026. The monthly change in the spread is hardly explained by the previous month's differentials with Germany (R2 1-16% by regime): nothing survives false-discovery control; the nominal best, industrial-growth gap (-0.025 pp per SD, q = 0.27 over the full sample and q = 0.16 since 2020), is small and does not hold across regimes. The same holds for Spain, France, Portugal and Greece (one coefficient in 80 has q < 0.05).
- **Business-cycle lead-lag (panel Granger tests, 27 countries).** Sentiment and industry confidence help predict industrial production growth one month ahead (and production predicts sentiment: both directions); at 3 months only industry confidence's *reverse* direction remains. Consumer confidence does not lead retail sales; retail sales lead consumer confidence at 3-6 months. Industrial production leads the change in unemployment only at 6 months, and sentiment and services confidence relate to unemployment in both directions at every lag. GDP growth leads employment and hours growth by 1-2 quarters (employment also predicts GDP at 2-4 quarters).
- **Italy's timing.** For Italy, the strongest prewhitened relation between industrial production growth and the change in unemployment is positive at lag 0 (r = 0.34, 2nd of 27 countries) because of 2020, when measured Italian unemployment fell in the lockdown months while output collapsed. Excluding March 2020 to June 2021 it becomes r = -0.07 at lag 0 and the strongest relation is a lead of production over unemployment of 6 months (r = -0.16), the same lead as the panel. Italy ranks 6th-13th of 26-27 on the sentiment and confidence relations, so nothing singles it out.
- **Wage pass-through.** Over 2005Q1-2021Q2 a one-point rise in average inflation (current and four previous quarters) goes with +0.44 pp of hourly labour-cost growth (p = 0.04, 28 countries); Italy's own estimate is 0.32, 19th of 27 (median 0.43). In 2021Q3-2023Q4, with quarter effects absorbing the common shock, the coefficient is 0.03 (p = 0.77): no detectable differential pass-through across countries in the energy-shock window.
- **Events (57 dated events x 4 series = 218 tests).** 16 have p < 0.05 (7%, close to the 5% expected by chance) and none has q < 0.10. The largest moves coincide with crisis turning points (Monti and the Salva Italia decree with unemployment +1.6 pp and the spread +1.3 pp over the following six months; the Draghi government and PNRR with sentiment +23 points, a post-pandemic rebound; the July 2022 ECB hike and TPI with the 10-year yield +1.8 pp), so the event dates cannot be separated from the cycle. No event type stands out (6-11% with p < 0.05 for policy, government and ECB events).
- **Limits:** associations and predictive order only; events are few, cluster in time and several share a month; HAC and country-clustered errors with 27 countries; seasonal adjustment by Eurostat; HICP from a reclassified dataset; GBR/USA/JPN are mostly absent from the monthly yield series."""))
