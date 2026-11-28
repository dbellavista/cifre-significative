"""Cells of notebook 06_hours (R3: how much of the fall in income per recipient can be hours, not pay)."""
from . import NOTEBOOKS, C, M

NOTEBOOKS["06_hours"] = [
    (M, """# Hours versus pay: INPS private-sector employees in Forlì-Cesena (`2008–2025`)

A reader of post 01 noted that the MEF income per recipient (−1.68% real, 2008–2024) puts a twelve-month full-time job and a part-time or three-month contract on the same footing, and that part-time grew in Italy after 2008. The MEF tables show neither hours nor contracts. This notebook uses a second source to bound the order of magnitude.

**Data:** INPS *Osservatorio sui lavoratori dipendenti* (private non-agricultural employees), by **province of work**: workers with at least one paid day in the year, annual pay, paid days, and the cut by `Presenza tempo parziale nell'anno` (`Si` = at least one part-time relationship in the year, not only part-time). The part-time cut exists for `2008–2018`, `2024` and `2025` only (the `2019–2023` blocks, published by INPS on 2026-11-24, carry no `No`/`Si`), so the part-time comparison uses the endpoints 2024 and 2025 (2025 is classified ATECO 2025, irrelevant for province totals). Totals of workers, pay and paid days exist for every year `2008–2025`; the 2021 values are those revised on 2026-11-24 (section 6).

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
    (M, """The line breaks between 2018 and 2024 are the years `2019–2023` without the part-time cut, not a series with a gap filled in.

## 2. Mean pay per worker: change of composition versus change within groups

Mean pay per worker `w = (1−s)·m_No + s·m_Si` (`s` = share of workers with part-time in the year). Between two years `within` is the change of `w` at constant share, `mix` is what the change of the share alone does (negative = drag), and `(1+within)(1+mix) = 1+total` exactly. All nominal and ratio-based, so no deflator is involved."""),
    (C, '''cols = ["share_pt_0", "share_pt_1", "share_pt_change_pp", "workers_change_pct", "days_per_worker_change_pct", "pay_per_day_change_pct", "pay_per_worker_change_pct", "within_pct", "mix_pct"]
out = {}
for y1 in (2024, 2018, 2025):
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

Endpoint 2018 or 2025 instead of 2024 (above), and the provinces whose territory changed (Milano and Monza, Ascoli Piceno and Fermo, Bari, Foggia and Barletta-Andria-Trani, the Sardinian provinces) kept in instead of dropped. Mean instead of median of the others."""),
    (C, '''rows = []
for y1 in (2024, 2018, 2025):
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
    (M, """## 6. Totals 2019–2025 (no part-time cut)

For `2019–2023` the Osservatorio gives only workers, pay and paid days by province (the part-time cut is missing), so it cannot extend sections 1–4; the totals are enough for the post-pandemic years. Growth is nominal; 2020 is the pandemic year. Provinces with a section hidden by the privacy threshold in the Open Data check (Arezzo, Matera) are dropped. The Open Data dataflow `DFB_ST_DIP_ATECO07_PROV_ANAG_01` agrees with the Osservatorio to the unit on 2014–2020 (except a masked cell) and is only a cross-check: for **2021 INPS revised the Osservatorio** on 2026-11-24 (Forlì-Cesena 122,340 workers, 122,085 in Open Data), and the Osservatorio is used."""),
    (C, '''tot = h.totals_panel(inps.load_totals())
display(tot.xs(h.UNIT, level="province").round(1))
rows = []
for y0, y1 in ((2019, 2020), (2019, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025), (2021, 2025), (2019, 2025)):
    g = h.totals_growth(tot, y0, y1, h.comparable_provinces(tot, (y0, y1)))
    r = h.unit_vs_others(g)[["unit", "median_others", "rank", "n_others"]]
    rows.append(r.assign(window=f"{y0}-{y1}").reset_index(names="measure"))
display(pd.concat(rows).set_index(["window", "measure"]).round(2))'''),
    (M, """In 2020 Forlì-Cesena lost more workers than the median of the other provinces and fewer paid days per worker; from 2021 workers are above 2019 and keep growing in line with the others. Pay per worker grows less than the median of the others in 2021–2022 and more in 2023–2024. Descriptive: the source does not say whether days were lost to short-time work, closures or contract changes. Claims below are computed in `analysis/claims.py`."""),
    (M, """## 7. Part-time by working-time type, 2008–2025 (monthly observatory, mean of the twelve months; one measure over the whole series)

The monthly Osservatorio gives, by province, the **number of workers present in the month** by working-time type (full time, horizontal, vertical and mixed part-time), but not pay or days. It covers the years without the annual part-time cut and, on the endpoints, can be checked against it. Files: `tipo_orario_media_<year>` (annual mean only, filter `Mese: Media`; from 2021 the site blocks the month-by-month table for statistical confidentiality; the 2019–2020 month-by-month tables, checked, have the same mean). Years loaded: 2008–2015 and 2017–2025; **2016 is missing** (the INPS page is in error), shown as a gap in the figure. The column order differs between exports (type first or province first): the parser reads it from the header. **This is not the annual `Si/No` of sections 1–4:** a worker counts once for each month present, so the mean is below the annual count (Forlì-Cesena 2019: 99,717 against 120,305), and the share is the share of the stock in a month, not of anyone with a part-time relationship in the year. Levels of the two shares must not be compared; the changes can be read on their own. The four types add up to the total within 2 workers (rounding of the monthly means)."""),
    (C, '''wt = inps.load_working_time()
wtp = h.working_time_panel(wt)
fcw = wtp.xs(h.UNIT, level="province")
medw = wtp.drop(index=h.UNIT, level="province").groupby("year")[["part_time_share_pct", "share_horizontal_pct", "share_vertical_pct", "share_mixed_pct"]].median()
display(pd.concat({"Forlì-Cesena": fcw[["workers_month", "part_time_share_pct", "share_horizontal_pct", "share_vertical_pct", "share_mixed_pct"]], "median others": medw}, axis=1).round(2))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(fcw.index, fcw.part_time_share_pct, "o-", color=ITALY_C, label="Forlì-Cesena")
ax.plot(medw.index, medw.part_time_share_pct, "s--", color=PEER_C, label="median of the other provinces")
ax.set_title("Part-time share of workers, mean of the twelve months (%)"); ax.legend(); ax.set_xlim(2007, 2026)
plt.tight_layout(); plt.show()
rows = []
for y0, y1 in ((2008, 2025), (2008, 2019), (2019, 2025), (2021, 2025)):
    r = h.unit_vs_others(h.working_time_change(wtp, y0, y1, h.comparable_provinces(wtp, (y0, y1))))[["unit", "median_others", "rank", "n_others"]]
    rows.append(r.assign(window=f"{y0}-{y1}").reset_index(names="measure"))
display(pd.concat(rows).set_index(["window", "measure"]).round(2))'''),
    (M, """Over the whole series the monthly part-time share of Forlì-Cesena rises from 18.7% in 2008 to 25.7% in 2025, +7.01 points against +7.82 for the median of the other provinces (#55 of 95), almost the same as the annual `Si` share of sections 1–2 (+6.99 points, 2008–2025): two different measures agree on the change. The path has two phases and no step: the share rises every year from 2008 to about 27% in 2017–2019 (the median of the others to about 30%), then it is flat or slightly lower: 26.9% in 2019, 25.7% in 2025, a fall of 1.14 points, a fall of 1.14 points against −1.52 for the median of the other provinces (#34 of 95); the fall is common, the province is in the middle. The rise before 2019 is gradual in both lines (2016 is missing, so the 2015–2017 step of about 1.6 points in Forlì-Cesena spans two years). It does not contradict the rise of the annual share between 2008 and 2025 (sections 1–2): the two shares measure different things and cover different years. Descriptive; horizontal part-time is about nine tenths of part-time in every year."""),
    (M, """## 8. Real pay: composition and the two groups (2008–2024 and 2008–2018)

Sections 1–4 are nominal. Here pay is deflated with the national price index used in post 01 (`reports/income_evidence/deflator_audit.csv`, column `panel_chained_2020_1`), the same for every province, so ranks do not change; only the levels of the changes move. `within_real` is the mean at the 2008 part-time share; `(1+within_real)(1+mix) = 1+total_real`. The index stops at 2024, so 2025 is not deflated."""),
    (C, '''prices = pd.read_csv("studies/cesena/reports/income_evidence/deflator_audit.csv").set_index("year").panel_chained_2020_1
for y1 in (2024, 2018):
    rr = h.real_pay_changes(panel, 2008, y1, prices, h.comparable_provinces(panel, (2008, y1)))
    er = [p for p in h.ER_PROVINCES if p in rr.index]
    print(f"\\n--- 2008-{y1}, real, {len(rr)} provinces ---")
    display(pd.concat({"all provinces": h.unit_vs_others(rr)[["unit", "median_others", "rank", "n_others"]],
                       "Emilia-Romagna": h.unit_vs_others(rr, others=er)[["median_others", "rank"]]}, axis=1).round(2))'''),
    (M, """From 2008 to 2024 real mean pay per worker in Forlì-Cesena changes −6.02% (median of the other provinces −8.64%, #20 of 95). At the 2008 part-time share it would be −1.88% (others −2.61%): most of the INPS fall is composition. Workers with no part-time relationship in the year, whose paid days barely move (254.7 and 254.1), still lose −2.38% (others −3.14%, #40 of 95), so the fall is not only hours. The loss is all after 2018: from 2008 to 2018 the same group gains +3.88% (others +4.04%). Descriptive; national prices, provinces of work, private employees."""),
    (M, """## Takeaways

*Descriptive, not causal; INPS private non-agricultural employees by province of work, 2008, 2024 and 2025 for the part-time cut; nominal except item 6; 2019–2023 only as totals (the cut is missing there); 2021 as revised by INPS on 2026-11-24.*

1. **Part-time grew in Forlì-Cesena, as elsewhere.** The share of workers with at least one part-time relationship in the year goes from 24.4% to 31.6% (+7.22 points), against +8.49 points for the median of the other provinces (#56 of 95): the province is not an outlier.
2. **Paid days per worker barely moved:** 244.7 in 2008, 244.4 in 2024 (−0.13%; median of the others +0.75%). Days understate the effect of part-time, because a part-time day is paid much less than a full-time day: pay per paid day of workers with a part-time job is about 56% of the others, in both years.
3. **Composition lowers the mean.** At the pay of each group in 2024, the rise of the part-time share alone puts the mean pay per worker about 4.22% below what the 2008 share would give (median of the other provinces −5.12%; median of the other Emilia-Romagna provinces −3.95%, Forlì-Cesena #7 of 9). The rest, `within`, is +32.34% (nominal), against +31.35%.
4. **Reading for post 01:** a composition effect of about four percent in this source is of the same order of magnitude as the −1.68% real fall of MEF employment income per recipient, so the reader's mechanism is plausible and not negligible. It is not an attribution: the two universes differ (workplace versus residence, private employees versus MEF's broader item with co.co.co., grants and benefits), part-time here means at least one part-time relationship in the year, and days are not hours. Forlì-Cesena looks like the other provinces on every measure of this notebook.
5. **Robustness:** with the 2018 endpoint the composition effect is −5.14% (others −6.38%); with the 2025 endpoint it is −4.08% (others −4.83%; share +6.99 points against +8.29, #59 of 95); keeping the provinces with changed territory the median of the others is −5.30%; the picture does not change.
6. **Real pay (section 8, national price index of post 01):** from 2008 to 2024 the real mean pay per worker changes −6.02% (others −8.64%); at the 2008 part-time share −1.88% (others −2.61%); workers with no part-time in the year −2.38% (others −3.14%, #40 of 95), with the same paid days. Composition is the larger part of the INPS fall, but a real fall remains without it, and it comes after 2018 (2008–2018: +3.88% for the same group).
7. **Totals 2019–2025:** in 2020 Forlì-Cesena loses 3.10% of workers (median of the others −1.54%) and 6.05% of paid days per worker (others −8.57%, #6 of 95); in 2021, against 2019, workers are +1.69% and paid days per worker −1.38% (others −2.60%), pay per worker +1.37% nominal (others −0.09%, #11 of 95). From 2021 to 2025 workers are +10.50% (others +9.86%), paid days per worker +3.19% (others +4.71%, #73 of 95), pay per paid day +10.30% (others +9.78%) and pay per worker +13.82% (others +15.41%, #68 of 95). The 2021–2024 jump is no longer a gap: 2022 is +2.70% on pay per worker (others +4.13%), 2023 +3.10% (others +3.48%), 2024 +4.08% (others +3.62%). Forlì-Cesena stays near the middle of the provinces.

**Limits.** Paid days are not hours; a worker with several jobs counts once; the INPS cut `Si` includes full-time workers with a part-time job in the year; the part-time cut is missing for 2019–2023; the 2025 data use ATECO 2025 (no effect on province totals, a break for any sector cut); the 2021 Open Data copy differs from the revised Osservatorio. The conclusion for readers is bounded: part-time plausibly weighs several percent on mean annual pay per worker; it does not say how much of the −1.68% it accounts for."""),
]
