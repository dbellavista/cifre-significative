"""Cells of notebook 06_hours (R3: how much of the fall in income per recipient can be hours, not pay)."""
from . import NOTEBOOKS, C, M

NOTEBOOKS["06_hours"] = [
    (M, """# Hours versus pay: INPS private-sector employees in Forlì-Cesena (`2008–2024`)

A reader of post 01 noted that the MEF income per recipient (−1.68% real, 2008–2024) puts a twelve-month full-time job and a part-time or three-month contract on the same footing, and that part-time grew in Italy after 2008. The MEF tables show neither hours nor contracts. This notebook uses a second source to bound the order of magnitude.

**Data:** INPS *Osservatorio sui lavoratori dipendenti* (private non-agricultural employees), by **province of work**: workers with at least one paid day in the year, annual pay, paid days, and the cut by `Presenza tempo parziale nell'anno` (`Si` = at least one part-time relationship in the year, not only part-time). Years `2008–2018` and `2024`: **`2019–2023` could not be retrieved** from the INPS site (empty error); the comparison uses the two endpoints.

**Limits (stated before the results):**
- Paid days are not hours: horizontal part-time counts whole days, so days per worker understate any hours effect.
- The universe is not the MEF one: INPS is workplace, province, private non-agricultural employees; the MEF item is residence, municipality, and includes co.co.co., grants and replacement benefits.
- A worker with several jobs counts once, classified by the last job. Descriptive only, not causal."""),
    (C, '''from cifresignificative.core.sources import inps_osservatorio as inps
from cifresignificative.studies.cesena.analysis import hours as h

raw = inps.load()
panel = h.province_panel(raw)
print(f"rows {len(raw)}; years {sorted(raw.year.unique())}; provinces {raw[raw.kind == 'province'].province.nunique()}")
keep = h.comparable_provinces(panel, (2008, 2024))
keep_all = h.comparable_provinces(panel, (2008, 2024), strict=False)
print(f"comparable provinces 2008-2024: {len(keep)} (strict, territory unchanged), {len(keep_all)} (all present in both years); dropped as redrawn: {sorted(set(keep_all) - set(keep))}")'''),
    (M, """## 1. Forlì-Cesena: share of workers with part-time, paid days and pay per day"""),
    (C, '''fc = panel.xs(h.UNIT, level="province")
oth = panel.drop(index=h.UNIT, level="province")
oth = oth[oth.index.get_level_values("province").isin(h.comparable_provinces(panel, (2008, 2024)))]
med = oth.groupby("year")[["share_pt_pct", "days_per_worker", "pay_per_day", "pt_pay_per_day_ratio"]].median()
tab = pd.DataFrame({
    "workers": fc.workers, "share_pt_pct": fc.share_pt_pct, "share_pt_median_others": med.share_pt_pct,
    "days_per_worker": fc.days_per_worker, "days_per_worker_median_others": med.days_per_worker,
    "days_pw_no": fc.days_per_worker_no, "days_pw_si": fc.days_per_worker_si,
    "pay_per_day_no": fc.pay_per_day_no, "pay_per_day_si": fc.pay_per_day_si, "pt_to_ft_pay_per_day": fc.pt_pay_per_day_ratio,
})
display(tab.round(2))'''),
    (C, '''fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(fc.index, fc.share_pt_pct, "o-", color=ITALY_C, label="Forlì-Cesena")
ax[0].plot(med.index, med.share_pt_pct, "s--", color=PEER_C, label="median of the other provinces")
ax[0].set_title("Workers with a part-time job in the year (%)"); ax[0].legend(); ax[0].set_xlim(2007, 2025)
ax[1].plot(fc.index, fc.days_per_worker, "o-", color=ITALY_C, label="Forlì-Cesena")
ax[1].plot(med.index, med.days_per_worker, "s--", color=PEER_C, label="median of the other provinces")
ax[1].set_title("Paid days per worker"); ax[1].legend(); ax[1].set_xlim(2007, 2025)
plt.tight_layout(); plt.show()'''),
    (M, """The line breaks between 2018 and 2024 are the missing years `2019–2023`, not a series with a gap filled in.

## 2. Mean pay per worker: change of composition versus change within groups

Mean pay per worker `w = (1−s)·m_No + s·m_Si` (`s` = share of workers with part-time in the year). Between two years `within` is the change of `w` at constant share, `mix` is what the change of the share alone does (negative = drag), and `(1+within)(1+mix) = 1+total` exactly. All nominal and ratio-based, so no deflator is involved."""),
    (C, '''cols = ["share_pt_0", "share_pt_1", "share_pt_change_pp", "workers_change_pct", "days_per_worker_change_pct", "pay_per_day_change_pct", "pay_per_worker_change_pct", "within_pct", "mix_pct"]
out = {}
for y1 in (2024, 2018):
    dec = h.mix_decomposition(panel, 2008, y1, h.comparable_provinces(panel, (2008, y1)))
    out[y1] = dec
    print(f"\\n--- 2008-{y1}: {len(dec)} provinces (territory unchanged) ---")
    display(pd.concat({"Forlì-Cesena": dec.loc[[h.UNIT], cols].iloc[0], "median others": dec.drop(index=h.UNIT)[cols].median()}, axis=1).round(2))
    display(h.unit_vs_others(dec)[["unit", "median_others", "rank", "n_others", "share_others_below_pct"]].round(2))'''),
    (M, """## 3. Placebo ranking: where Forlì-Cesena sits among all provinces, then among Emilia-Romagna"""),
    (C, '''dec = out[2024]
er = [p for p in h.ER_PROVINCES if p in dec.index]
print("Emilia-Romagna provinces:", er)
display(dec.loc[er, ["share_pt_change_pp", "days_per_worker_change_pct", "pay_per_day_change_pct", "within_pct", "mix_pct", "pay_per_worker_change_pct"]].round(2))
display(h.unit_vs_others(dec, others=er)[["unit", "median_others", "rank", "n_others", "share_others_below_pct"]].round(2).loc[["share_pt_change_pp", "days_per_worker_change_pct", "within_pct", "mix_pct"]])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(dec.share_pt_change_pp, dec.mix_pct, s=14, color="lightgray", label="other provinces")
ax.scatter(dec.loc[er].share_pt_change_pp, dec.loc[er].mix_pct, s=28, color=PEER_C, label="Emilia-Romagna")
ax.scatter(*dec.loc[h.UNIT, ["share_pt_change_pp", "mix_pct"]], s=70, color=ITALY_C, label=h.UNIT, zorder=3)
ax.set_xlabel("change of the share with part-time, 2008–2024 (points)"); ax.set_ylabel("effect on mean pay per worker (%)"); ax.legend()
plt.tight_layout(); plt.show()'''),
    (M, """## 4. Robustness

Endpoint 2018 instead of 2024 (above), and the provinces whose territory changed (Milano and Monza, Ascoli Piceno and Fermo, Bari, Foggia and Barletta-Andria-Trani, the Sardinian provinces) kept in instead of dropped. Mean instead of median of the others."""),
    (C, '''rows = []
for y1 in (2024, 2018):
    for strict in (True, False):
        d_ = h.mix_decomposition(panel, 2008, y1, h.comparable_provinces(panel, (2008, y1), strict))
        o = d_.drop(index=h.UNIT)
        rows.append({"window": f"2008-{y1}", "territory unchanged only": strict, "n": len(d_),
                     "mix FC": d_.loc[h.UNIT, "mix_pct"], "mix median others": o.mix_pct.median(), "mix mean others": o.mix_pct.mean(),
                     "share change FC (pp)": d_.loc[h.UNIT, "share_pt_change_pp"], "share change median others": o.share_pt_change_pp.median()})
display(pd.DataFrame(rows).round(2))'''),
    (M, """## 5. Claims computed from the data

The figures quoted in the takeaways are computed in `analysis/claims.py` (`_hours`, run `uv run python -m cifresignificative.studies.cesena.analysis.claims` after changing the data) and shown below."""),
    (C, '''from cifresignificative.studies.cesena.analysis.claims import claims_for_notebook

display(claims_for_notebook("06"))'''),
    (M, """## Takeaways

*Descriptive, not causal; INPS private non-agricultural employees by province of work, 2008 and 2024 (the years 2019–2023 could not be retrieved).*

1. **Part-time grew in Forlì-Cesena, as elsewhere.** The share of workers with at least one part-time relationship in the year goes from 24.4% to 31.6% (+7.22 points), against +8.49 points for the median of the other provinces (#56 of 95): the province is not an outlier.
2. **Paid days per worker barely moved:** 244.7 in 2008, 244.4 in 2024 (−0.13%; median of the others +0.75%). Days understate the effect of part-time, because a part-time day is paid much less than a full-time day: pay per paid day of workers with a part-time job is about 56% of the others, in both years.
3. **Composition lowers the mean.** At the pay of each group in 2024, the rise of the part-time share alone puts the mean pay per worker about 4.22% below what the 2008 share would give (median of the other provinces −5.12%; median of the other Emilia-Romagna provinces −3.95%, Forlì-Cesena #7 of 9). The rest, `within`, is +32.34% (nominal), against +31.35%.
4. **Reading for post 01:** a composition effect of about four percent in this source is of the same order of magnitude as the −1.68% real fall of MEF employment income per recipient, so the reader's mechanism is plausible and not negligible. It is not an attribution: the two universes differ (workplace versus residence, private employees versus MEF's broader item with co.co.co., grants and benefits), part-time here means at least one part-time relationship in the year, and days are not hours. Forlì-Cesena looks like the other provinces on every measure of this notebook.
5. **Robustness:** with the 2018 endpoint the composition effect is −5.14% (others −6.38%); keeping the provinces with changed territory the median of the others is −5.30%; the picture does not change.

**Limits.** Paid days are not hours; a worker with several jobs counts once; the INPS cut `Si` includes full-time workers with a part-time job in the year; years 2019–2023 are missing. The conclusion for readers is bounded: part-time plausibly weighs several percent on mean annual pay per worker; it does not say how much of the −1.68% it accounts for."""),
]
