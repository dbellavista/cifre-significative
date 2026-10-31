"""Cells of notebook 19_pension_reform_event_study."""
from . import NOTEBOOKS, C, M

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

NOTEBOOKS["19_pension_reform_event_study"].append((M, """## Takeaways (suggestive, not causal)
- **Older-worker employment rose far more in Italy than in controls.** Relative to 2008-2010, employment of 55-64 year-olds in 2015-2019 was 5.8 pp higher than in the control group (placebo median +0.2 pp; Italy 3rd largest rise of about 26 countries) and 60-64 year-olds 8.2 pp (placebo median +0.5; 5th, or 3rd excluding other reforming countries). The effective male exit age rose 0.6 years relative to controls (placebo median -0.1; 7th-8th), a smaller shift than employment.
- **Robust to the choice of controls:** excluding Spain, Greece, Portugal, France, the Netherlands and Germany (which also raised retirement ages in 2010-2015) leaves the 55-64 and 60-64 results essentially unchanged.
- **Youth outcomes cannot be attributed:** Italy's youth unemployment is 13.4 pp and the NEET rate 4.4 pp higher than the controls' relative to 2008-2010, but the 2011-2013 recession hit Italy exactly when the reform took effect, so the event study cannot separate the two (the literature's local-labour-market estimates of a youth-hiring spillover use firm-level variation).
- Caveats: pre-reform trends differ by country (the 2009-2010 reforms already raised retirement ages), the controls' own reforms are not fully excluded, and with 26 countries placebo ranks are a rough guide."""))
