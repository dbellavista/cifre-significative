"""Every figure quoted in `studies/cesena/reports/findings.md`, computed from the processed tables (`studies.cesena.analysis.claims`).

The 2026-10-30 review found narrative numbers that matched no result because they were typed by hand. Here each quoted figure is one row (`id`, `value`, `text` as printed, optional `rank`/`n`, `window`, `source`, `notebook`), a claim with a rank also yields `<id>_rank`, and `tests/studies/cesena/test_claims.py` fails when the text disagrees or a figure has no tag.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from cifresignificative.core.geo import CESENA, CORE_PEER_IDS
from cifresignificative.studies.cesena.analysis.economy import sector_size_swing
from cifresignificative.studies.cesena.analysis.fiscal import (
    compare_fiscal_regimes,
    mission_allocation_summary,
)
from cifresignificative.studies.cesena.analysis.income_followup import component_changes
from cifresignificative.studies.cesena.analysis.investments import (
    summarize_investments_by_comune,
)

PROCESSED_DIR = Path("data/processed/cesena")
REVIEW_GROWTH = Path("studies/cesena/reports/income_evidence/review_growth.csv")
CORE = [c for c in CORE_PEER_IDS if c != CESENA]


def pct(v: float, d: int = 2) -> str:
    return f"{v:+.{d}f}%"


def share(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}%"


def pp(v: float, d: int = 2) -> str:
    return f"{v:+.{d}f} pp"


def num(v: float, d: int = 1) -> str:
    return f"{v:,.{d}f}"


def signed(v: float, d: int = 1) -> str:
    return f"{v:+,.{d}f}"


def eur(v: float, d: int = 1) -> str:
    return f"€{v:,.{d}f}"


def meur(v: float, d: int = 1) -> str:
    return f"€{v / 1e6:,.{d}f}M"


def beur(v: float, d: int = 2) -> str:
    return f"€{v / 1e9:,.{d}f}B"


class _Claims:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, cid: str, value: float, text: str, nb: str, window: str, source: str, rank: int | None = None, n: int | None = None) -> None:
        self.rows.append({"id": cid, "value": float(value), "text": text, "rank": rank, "n": n, "window": window, "source": source, "notebook": nb})
        if rank is not None and n is not None:
            self.rows.append({"id": f"{cid}_rank", "value": float(rank), "text": f"#{int(rank)} of {int(n)}", "rank": rank, "n": n, "window": window, "source": source, "notebook": nb})


def _rank(ser: pd.Series, cid: str = CESENA) -> tuple[int, int]:
    ser = ser.dropna()
    return int((ser > ser.loc[cid]).sum() + 1), len(ser)


def _fiscal(c: _Claims, p: Path) -> None:
    fiscal, missions, demo = pd.read_parquet(p / "fiscal.parquet"), pd.read_parquet(p / "missions.parquet"), pd.read_parquet(p / "demography.parquet")
    reg = compare_fiscal_regimes(fiscal, core_only=True).set_index(["regime", "metric"])
    src = "fiscal.parquet"

    def line(cid: str, regime: str, metric: str, window: str) -> None:
        r = reg.loc[(regime, metric)]
        c.add(cid, r["cesena_mean"], eur(r["cesena_mean"], 0), "01", window, src, int(r["rank_1_to_n"]), int(r["n_comuni"]))
        c.add(f"{cid}_peer", r["peer_median"], eur(r["peer_median"], 0), "01", window, src)

    line("current_0815", "dpr194", "spese_correnti_comp_real_pc", "2008-2015")
    line("current_1621", "dlgs118", "spese_correnti_comp_real_pc", "2016-2021")
    line("capital_1621", "dlgs118", "spese_capitale_comp_real_pc", "2016-2021")
    line("loans_1621", "dlgs118", "accensione_mutui_comp_real_pc", "2016-2021")
    line("grants_1621", "dlgs118", "contributi_investimenti_comp_real_pc", "2016-2021")
    line("sales_1621", "dlgs118", "alienazioni_comp_real_pc", "2016-2021")
    line("fpv_1621", "dlgs118", "fpv_capitale_comp_real_pc", "2016-2021")
    line("taxes_1621", "dlgs118", "imposte_tasse_comp_real_pc", "2016-2021")
    cons = fiscal[(fiscal["budget_type"] == "consuntivo") & (fiscal["comune_id"] == CESENA)].set_index("year")["spese_capitale_comp_real_pc"]
    c.add("capital_2019", cons[2019], eur(cons[2019], 0), "01", "2019", src)
    ex19 = cons[[y for y in range(2016, 2022) if y != 2019]].mean()
    c.add("capital_1621_ex2019", ex19, eur(ex19, 0), "01", "2016-2021 without 2019", src)
    r = reg.loc[("dlgs118", "rigidity_official_pct")]
    c.add("rigidity_1618", r["cesena_mean"], share(r["cesena_mean"]), "01", "2016-2018", src, int(r["rank_1_to_n"]), int(r["n_comuni"]))
    c.add("rigidity_1618_peer", r["peer_median"], share(r["peer_median"]), "01", "2016-2018", src)

    # Waste booking: a city that keeps the waste tax in its budget also carries the waste service in current spending. The accounts itemise the tax in 2016-2018 only.
    early = fiscal[(fiscal["regime"] == "dlgs118") & (fiscal["budget_type"] == "consuntivo") & fiscal["year"].between(2016, 2018)]
    for col, tag in (("spese_correnti_comp_real_pc", "current"), ("imposte_tasse_comp_real_pc", "taxes")):
        net = (early[col] - early["tari_tarsu_comp_real_pc"]).groupby(early["comune_id"]).mean()
        rk, n = _rank(net)
        c.add(f"{tag}_x_1618", net[CESENA], eur(net[CESENA], 0), "01", "2016-2018, without the waste tax", src, rk, n)
        c.add(f"{tag}_x_1618_peer", net[CORE].median(), eur(net[CORE].median(), 0), "01", "2016-2018, without the waste tax", src)
    rig = early.pivot(index="comune_id", columns="year", values="rigidity_official_pct")
    c.add("rigidity_faenza_2016", rig.loc["039010", 2016], share(rig.loc["039010", 2016]), "01", "2016, Faenza, as in the source", src)
    rk, n = _rank(rig.drop(index="039010").mean(axis=1))
    c.add("rigidity_1618_x_faenza", rig.loc[CESENA].mean(), share(rig.loc[CESENA].mean()), "01", "2016-2018, Faenza left out", src, rk, n)

    m = mission_allocation_summary(missions, demo, core_only=True).set_index("mission_code")
    for code in ("M12", "M03", "M05"):
        c.add(f"{code.lower()}_current", m.loc[code, "cesena_correnti_real_pc"], eur(m.loc[code, "cesena_correnti_real_pc"], 0), "01", "2016-2021", "missions.parquet")
        c.add(f"{code.lower()}_current_peer", m.loc[code, "peer_correnti_real_pc"], eur(m.loc[code, "peer_correnti_real_pc"], 0), "01", "2016-2021", "missions.parquet")
    for code in ("M01", "M10"):
        c.add(f"{code.lower()}_capital", m.loc[code, "cesena_capitale_real_pc"], eur(m.loc[code, "cesena_capitale_real_pc"], 0), "01", "2016-2021", "missions.parquet")
        c.add(f"{code.lower()}_capital_peer", m.loc[code, "peer_capitale_real_pc"], eur(m.loc[code, "peer_capitale_real_pc"], 0), "01", "2016-2021", "missions.parquet")

    panel = pd.read_parquet(p / "panel_comuni.parquet")
    s = panel[panel["year"] == 2024].set_index("comune_id")["mef_addizionale_real_pc"]
    rk, n = _rank(s)
    c.add("surcharge_2024", s[CESENA], eur(s[CESENA], 0), "01", "2024", "panel_comuni.parquet", rk, n)
    c.add("surcharge_2024_core", s[CORE].median(), eur(s[CORE].median(), 0), "01", "2024", "panel_comuni.parquet")

    # Consolidated perimeter
    sc = pd.read_parquet(p / "siope_consolidated.parquet")
    y = sc[sc["year"] == 2024].set_index("comune_id")
    src = "siope_consolidated.parquet"
    for cid, col in (("direct_current_2024", "direct_correnti_pc"), ("consolidated_current_2024", "consolidated_correnti_pc")):
        rk, n = _rank(y[col])
        c.add(cid, y.loc[CESENA, col], eur(y.loc[CESENA, col], 0), "01", "2024", src, rk, n)
        med = y.loc[CORE, col].median()
        c.add(f"{cid}_core", med, eur(med, 0), "01", "2024", src)
        c.add(f"{cid}_gap", (y.loc[CESENA, col] / med - 1) * 100, pct((y.loc[CESENA, col] / med - 1) * 100, 1), "01", "2024", src)
    all_med = y.drop(index=CESENA)["consolidated_correnti_pc"].median()
    c.add("consolidated_current_2024_all", all_med, eur(all_med, 0), "01", "2024", src)
    c.add("consolidated_current_2024_all_gap", (y.loc[CESENA, "consolidated_correnti_pc"] / all_med - 1) * 100, pct((y.loc[CESENA, "consolidated_correnti_pc"] / all_med - 1) * 100, 1), "01", "2024", src)
    c.add("transfer_to_unione_2024", y.loc[CESENA, "netted_correnti_eur"], meur(y.loc[CESENA, "netted_correnti_eur"], 2), "01", "2024", src)
    rk, n = _rank(y["consolidated_personale_pc"])
    c.add("consolidated_personnel_2024", y.loc[CESENA, "consolidated_personale_pc"], eur(y.loc[CESENA, "consolidated_personale_pc"], 0), "01", "2024", src, rk, n)
    c.add("consolidated_personnel_2024_core", y.loc[CORE, "consolidated_personale_pc"].median(), eur(y.loc[CORE, "consolidated_personale_pc"].median(), 0), "01", "2024", src)

    # Cash rigidity after the accrual series stops (2019-2024), with the 2017-2018 overlap as anchor
    cp = pd.read_parquet(p / "siope_cash_panel.parquet")
    cp = cp[cp["entity_type"] == "comune"]
    src = "siope_cash_panel.parquet"
    for col, tag, note in (("rigidity_cash_pct", "rigidity_cash", ""), ("rigidity_cash_ex_tari_pct", "rigidity_cash_x", " without waste tax")):
        mean = cp[cp["year"].between(2019, 2024)].groupby("comune_id")[col].mean()
        rk, n = _rank(mean)
        c.add(f"{tag}_1924", mean[CESENA], share(mean[CESENA]), "01", f"2019-2024, cash{note}", src, rk, n)
        c.add(f"{tag}_1924_peer", mean[CORE].median(), share(mean[CORE].median()), "01", f"2019-2024, cash{note}", src)
        yr = cp[cp["comune_id"] == CESENA].set_index("year")[col]
        for y in (2019, 2024):
            c.add(f"{tag}_{y}", yr[y], share(yr[y]), "01", f"{y}, cash{note}", src)
    tari = cp[cp["comune_id"] == CESENA].set_index("year")["tari_tarsu_cassa_eur"]
    for y in (2022, 2023):
        c.add(f"tari_cash_{y}", tari[y], meur(tari[y], 1), "01", str(y), src)
    y24 = cp[cp["year"] == 2024]
    low = int(((y24["tari_tarsu_cassa_eur"] / y24["entrate_correnti_cassa_eur"]) < 1 / 30).sum())
    c.add("tari_low_n", low, f"{low} of {len(y24)}", "01", "2024", src)
    # Consolidated current spending net of waste-tax cash receipts (a proxy for the cost of the waste service kept in the budget)
    sc24 = sc[sc["year"] == 2024].set_index("comune_id")
    tari_pc = y24.set_index("comune_id")["tari_tarsu_cassa_eur"] / sc24["pop_total"]
    net = sc24["consolidated_correnti_pc"] - tari_pc
    note = "2024, cash, net of waste-tax receipts"
    rk, n = _rank(net)
    c.add("consolidated_current_x_2024", net[CESENA], eur(net[CESENA], 0), "01", note, src, rk, n)
    for tag, med in (("core", net[CORE].median()), ("all", net.drop(index=CESENA).median())):
        c.add(f"consolidated_current_x_2024_{tag}", med, eur(med, 0), "01", note, src)
        c.add(f"consolidated_current_x_2024_{tag}_gap", (net[CESENA] / med - 1) * 100, pct((net[CESENA] / med - 1) * 100, 1), "01", note, src)
    booked = tari_pc.drop(index=CESENA)
    booked = booked[booked > 100]  # peers that still collect the waste tax through the budget
    c.add("tari_peers_booked_n", len(booked), f"{len(booked)} of {len(tari_pc) - 1}", "01", "2024, waste tax above 100 euro per resident", src)
    c.add("tari_peers_booked_lo", booked.min(), eur(booked.min(), 0), "01", "2024, per resident", src)
    c.add("tari_peers_booked_hi", booked.max(), eur(booked.max(), 0), "01", "2024, per resident", src)
    cur = cp[cp["comune_id"] == CESENA].set_index("year")["spese_correnti_cassa_eur"]
    for y in (2022, 2023):
        c.add(f"current_cash_{y}", cur[y], meur(cur[y], 1), "01", f"{y}, cash current spending, Comune alone", src)
    # Cesena accruals from the Rendiconto CSVs against the cash proxy for the same years (one city, anchor only)
    ra = p / "rendiconto_accrual.parquet"
    if ra.exists():
        acc_c = pd.read_parquet(ra).set_index("year")["rigidity_accrual_pct"]
        for y in (2022, 2023, 2024):
            c.add(f"rigidity_accrual_{y}", acc_c[y], share(acc_c[y]), "01", f"{y}, accrual", "rendiconto_accrual.parquet")
        cash_c = cp[cp["comune_id"] == CESENA].set_index("year")["rigidity_cash_pct"]
        gap = [abs(float(acc_c[y]) - float(cash_c[y])) for y in (2022, 2023, 2024)]
        c.add("rigidity_accrual_cash_gap_max", max(gap), f"{max(gap):.1f} points", "01", "2022-2024", "rendiconto_accrual.parquet + siope_cash_panel.parquet")
        if {2018, 2021} <= set(acc_c.index):  # old-site rendiconto CSVs (core.sources.comune_archive)
            for y in (2018, 2019, 2020, 2021):
                c.add(f"rigidity_accrual_{y}", acc_c[y], share(acc_c[y]), "01", f"{y}, accrual (old-site CSV)", "rendiconto_accrual.parquet")
            gap_old = [abs(float(acc_c[y]) - float(cash_c[y])) for y in (2018, 2019, 2020, 2021)]
            c.add("rigidity_accrual_cash_gap_max_18_21", max(gap_old), f"{max(gap_old):.1f} points", "01", "2018-2021", "rendiconto_accrual.parquet + siope_cash_panel.parquet")
            c.add("rigidity_accrual_change_21_25", acc_c[2025] - acc_c[2021], pp(acc_c[2025] - acc_c[2021], 1), "01", "2021 to 2025, accrual as published (the 2023 break in the waste levy is inside)", "rendiconto_accrual.parquet")
        if "rigidity_accrual_ex_tari_pct" in pd.read_parquet(ra).columns:  # the waste-tax levy (TARI, until 2022) taken out of the denominator
            ra_df = pd.read_parquet(ra).set_index("year")
            ex, levy = ra_df["rigidity_accrual_ex_tari_pct"], ra_df["tari_levy_eur"]
            c.add("tari_levy_2022", levy[2022], meur(levy[2022], 2), "01", "2022, accrued TARI levy (Scheda Tassa sui Rifiuti)", "rendiconto_accrual.parquet")
            if {2018, 2021} <= set(ex.dropna().index):  # levy sheets of the old-site Giunta reports
                for y in (2018, 2019, 2020, 2021):
                    c.add(f"rigidity_accrual_ex_tari_{y}", ex[y], share(ex[y]), "01", f"{y}, accrual without the TARI levy", "rendiconto_accrual.parquet")
                c.add("rigidity_accrual_ex_tari_change_18_21", ex[2021] - ex[2018], pp(ex[2021] - ex[2018], 1), "01", "2018 to 2021, accrual without the TARI levy", "rendiconto_accrual.parquet")
                c.add("rigidity_accrual_ex_tari_change_21_25", ex[2025] - ex[2021], pp(ex[2025] - ex[2021], 1), "01", "2021 to 2025, accrual without the TARI levy (2023 onwards has none to take out)", "rendiconto_accrual.parquet")
            c.add("rigidity_accrual_ex_tari_2022", ex[2022], share(ex[2022]), "01", "2022, accrual without the TARI levy", "rendiconto_accrual.parquet")
            c.add("rigidity_accrual_change_22_23", acc_c[2023] - acc_c[2022], pp(acc_c[2023] - acc_c[2022], 1), "01", "2022 to 2023, accrual as published", "rendiconto_accrual.parquet")
            c.add("rigidity_accrual_ex_tari_change_22_23", ex[2023] - ex[2022], pp(ex[2023] - ex[2022], 1), "01", "2022 to 2023, accrual without the TARI levy", "rendiconto_accrual.parquet")
            ex_cash = cp[cp["comune_id"] == CESENA].set_index("year")["rigidity_cash_ex_tari_pct"]
            c.add("rigidity_accrual_ex_tari_cash_gap_2022", abs(ex[2022] - ex_cash[2022]), f"{abs(ex[2022] - ex_cash[2022]):.1f} points", "01", "2022, accrual against cash, both without TARI", "rendiconto_accrual.parquet + siope_cash_panel.parquet")
    # Official indicators printed in the rendiconto Piano degli indicatori (Cesena only)
    ip = p / "indicatori_rendiconto.parquet"
    if ip.exists():
        ind = pd.read_parquet(ip).set_index(["year", "code"])["value"]
        for y in (2023, 2024):
            c.add(f"rigidity_official_ind_{y}", ind[(y, "1.1")], share(ind[(y, "1.1")]), "01", f"{y}, official indicator 1.1", "indicatori_rendiconto.parquet")
        for y in (2020, 2021, 2022):
            if (y, "1.1") in ind.index:
                c.add(f"rigidity_official_ind_{y}", ind[(y, "1.1")], share(ind[(y, "1.1")]), "01", f"{y}, official indicator 1.1 (old-site PDF)", "indicatori_rendiconto.parquet")
        c.add("personnel_pc_ind_2024", ind[(2024, "4.4")], eur(ind[(2024, "4.4")], 0), "01", "2024, official indicator 4.4", "indicatori_rendiconto.parquet")
        c.add("interest_incidence_2024", ind[(2024, "6.1")], share(ind[(2024, "6.1")]), "01", "2024, official indicator 6.1", "indicatori_rendiconto.parquet")
    # Conto annuale del personale: permanent staff and total labour cost of the municipality alone (Cesena only, nominal euro)
    cap = p / "conto_annuale.parquet"
    if cap.exists():
        ca = pd.read_parquet(cap).set_index("year")
        src = "conto_annuale.parquet"
        for y in (2020, 2024):
            c.add(f"conto_heads_{y}", ca.loc[y, "heads"], f"{ca.loc[y, 'heads']:.0f}", "01", f"{y}, 31 December", src)
            c.add(f"conto_cost_{y}", ca.loc[y, "labour_cost_eur"], meur(ca.loc[y, "labour_cost_eur"], 1), "01", f"{y}, nominal", src)
        c.add("conto_fte_2024", ca.loc[2024, "fte"], f"{ca.loc[2024, 'fte']:.0f}", "01", "2024, full-time equivalents", src)
        chg = (ca.loc[2024, "labour_cost_eur"] / ca.loc[2020, "labour_cost_eur"] - 1) * 100
        c.add("conto_cost_change_2024", chg, pct(chg, 1), "01", "2020-2024, nominal", src)
        # detail tables of the same PDFs: flexible staff, the decentralised-bargaining fund, mean pay per head
        fx, fd = p / "conto_annuale_flexible.parquet", p / "conto_annuale_funds.parquet"
        if fx.exists() and fd.exists():
            flex = pd.read_parquet(fx).pivot(index="year", columns="kind", values="cost_eur")
            flex_cost = flex[["fixed_term", "agency", "training"]].sum(axis=1)
            for y in (2020, 2024):
                c.add(f"conto_flex_share_{y}", flex_cost[y] / ca.loc[y, "labour_cost_eur"] * 100, share(flex_cost[y] / ca.loc[y, "labour_cost_eur"] * 100), "01", f"{y}, fixed-term + agency + training-contract pay over total labour cost", "conto_annuale_flexible.parquet")
            fund = pd.read_parquet(fd).query("fund == 'decentralised_fund'").set_index("year")["resources_eur"]
            for y in (2020, 2024):
                c.add(f"conto_fund_{y}", fund[y], meur(fund[y], 2), "01", f"{y}, decentralised fund resources (non-managerial staff), nominal", "conto_annuale_funds.parquet")
            c.add("conto_fund_change_2024", (fund[2024] / fund[2020] - 1) * 100, pct((fund[2024] / fund[2020] - 1) * 100, 1), "01", "2020-2024, nominal", "conto_annuale_funds.parquet")
    # staffing plan (DUP): personnel-spending limit table, plan against outturn, staff in service, retirements forecast
    fl, fs, fr = p / "fabbisogni_limits.parquet", p / "fabbisogni_staff.parquet", p / "fabbisogni_retirements.parquet"
    if fl.exists() and fs.exists() and fr.exists():
        from cifresignificative.studies.cesena.analysis.fabbisogni import (
            plan_vs_outturn,
        )

        lim, st, ret = pd.read_parquet(fl), pd.read_parquet(fs).set_index("ref_date"), pd.read_parquet(fr).set_index("year")
        out = lim[lim["outturn"]].sort_values("order").drop_duplicates("year", keep="last").set_index("year")
        plan = lim[~lim["outturn"] & (lim["source"].str.startswith("11454546"))].set_index("year")  # September 2025 DUP, budget years 2025-2028
        src = "fabbisogni_limits.parquet"
        for y in (2021, 2024):
            c.add(f"limit_total_{y}", out.loc[y, "total_eur"], meur(out.loc[y, "total_eur"], 2), "01", f"{y}, outturn, art. 1 c. 557 total incl. staff transferred to the Unione", src)
        c.add("limit_municipal_2024", out.loc[2024, "municipal_eur"], meur(out.loc[2024, "municipal_eur"], 2), "01", "2024, outturn, Comune alone", src)
        c.add("limit_unione_2024", out.loc[2024, "unione_eur"], meur(out.loc[2024, "unione_eur"], 2), "01", "2024, outturn, staff transferred to the Unione", src)
        for k, lab in (("municipal_eur", "Comune alone"), ("unione_eur", "staff transferred to the Unione"), ("total_eur", "total")):
            c.add(f"limit_plan_{k[:-4]}_2028", plan.loc[2028, k], meur(plan.loc[2028, k], 2), "01", f"2028, September 2025 DUP plan, {lab}", src)
        c.add("limit_plan_unione_2025", plan.loc[2025, "unione_eur"], meur(plan.loc[2025, "unione_eur"], 2), "01", "2025, September 2025 DUP plan, staff transferred to the Unione", src)
        for _, r in plan_vs_outturn(lim).iterrows():
            c.add(f"limit_gap_{int(r['year'])}", r["gap_pct"], pct(r["gap_pct"], 1), "01", f"{int(r['year'])}, outturn against the last published plan, total", src)
        for d_ in ("2024-09-30", "2025-09-30"):
            c.add(f"staff_dup_{d_[:4]}", st.loc[d_, "heads"], f"{st.loc[d_, 'heads']:.0f}", "01", f"permanent staff in service on {d_} (DUP table)", "fabbisogni_staff.parquet")
        r3 = ret.loc[2026:2028]
        c.add("retirements_2026_2028", r3["retirements"].sum(), f"{r3['retirements'].sum():.0f}", "01", "2026-2028 forecast retirements (September 2025 DUP)", "fabbisogni_retirements.parquet")
        c.add("retirements_saving_2026_2028", r3["saving_eur"].sum(), meur(r3["saving_eur"].sum(), 2), "01", "2026-2028 forecast saving from retirements", "fabbisogni_retirements.parquet")
    # Consolidated accounts of the Comune group (old transparency site): 2018, 2020 (prior-year column of the 2021 report) and 2021; 2019 is a scan read by OCR
    cs, cper = p / "consolidated_statements.parquet", p / "consolidated_perimeter.parquet"
    if cs.exists() and cper.exists():
        st = pd.read_parquet(cs).set_index(["year", "item"])

        def smeur(v: float, d: int = 1) -> str:
            return f"-{meur(-v, d)}" if v < 0 else meur(v, d)

        for y in (2018, 2019, 2020, 2021):
            c.add(f"group_result_{y}", st.loc[(y, "net_result"), "consolidated_eur"], smeur(st.loc[(y, "net_result"), "consolidated_eur"], 2), "01", f"{y}, group result before the minorities' share", "consolidated_statements.parquet")
        for y in (2018, 2021):
            c.add(f"comune_result_{y}", st.loc[(y, "net_result"), "capogruppo_eur"], smeur(st.loc[(y, "net_result"), "capogruppo_eur"], 2), "01", f"{y}, the Comune alone", "consolidated_statements.parquet")
        c.add("group_adjustments_2018", st.loc[(2018, "financial_adjustments"), "consolidated_eur"], smeur(st.loc[(2018, "financial_adjustments"), "consolidated_eur"], 2), "01", "2018, value adjustments of financial assets, group", "consolidated_statements.parquet")
        c.add("group_adjustments_2018_comune", st.loc[(2018, "financial_adjustments"), "capogruppo_eur"], smeur(st.loc[(2018, "financial_adjustments"), "capogruppo_eur"], 2), "01", "2018, value adjustments of financial assets, Comune alone", "consolidated_statements.parquet")
        c.add("group_adjustments_2018_impact", st.loc[(2018, "financial_adjustments"), "impact_eur"], smeur(st.loc[(2018, "financial_adjustments"), "impact_eur"], 2), "01", "2018, consolidation effect on the value adjustments", "consolidated_statements.parquet")
        c.add("group_inventories_2018", st.loc[(2018, "inventories"), "consolidated_eur"], smeur(st.loc[(2018, "inventories"), "consolidated_eur"], 1), "01", "2018, group inventories", "consolidated_statements.parquet")
        c.add("group_inventories_2019", st.loc[(2019, "inventories"), "consolidated_eur"], smeur(st.loc[(2019, "inventories"), "consolidated_eur"], 1), "01", "2019, group inventories (OCR of the scan)", "consolidated_statements.parquet")
        c.add("group_inventories_2020", st.loc[(2020, "inventories"), "consolidated_eur"], smeur(st.loc[(2020, "inventories"), "consolidated_eur"], 1), "01", "2020, group inventories", "consolidated_statements.parquet")
        c.add("group_equity_2021", st.loc[(2021, "equity"), "consolidated_eur"], meur(st.loc[(2021, "equity"), "consolidated_eur"], 0), "01", "2021, group equity", "consolidated_statements.parquet")
        c.add("group_cash_2021", st.loc[(2021, "cash"), "consolidated_eur"], meur(st.loc[(2021, "cash"), "consolidated_eur"], 1), "01", "2021, group cash", "consolidated_statements.parquet")
        per = pd.read_parquet(cper).groupby("year").size()
        c.add("group_perimeter_2018", per[2018], f"{per[2018]}", "01", "bodies in the 2018 consolidation perimeter", "consolidated_perimeter.parquet")
        c.add("group_perimeter_2021", per[2021], f"{per[2021]}", "01", "bodies in the 2021 consolidation perimeter", "consolidated_perimeter.parquet")
    # controlled companies: crisis-risk report (printed alarms and ratio alarms from the thresholds in the row labels)
    rk = p / "controlled_risk.parquet"
    if rk.exists():
        risk = pd.read_parquet(rk)
        printed = risk[["negative_operations_3y", "losses_over_15pct", "going_concern_doubt"]].any(axis=1)
        c.add("risk_companies", risk["company"].nunique(), f"{risk['company'].nunique()}", "01", "controlled companies in the 2023 and 2025 risk reports", "controlled_risk.parquet")
        c.add("risk_printed_alarms", printed.sum(), f"{printed.sum()}", "01", "company-years with any of the three printed alarms, 2023 and 2025", "controlled_risk.parquet")
        f25 = risk[(risk["company"] == "Filiera Ortofrutticola Romagnola") & (risk["year"] == 2025)].iloc[0]
        c.add("risk_filiera_liquidity_2025", f25["liquidity_index"], f"{f25['liquidity_index']:.2f}", "01", "2025 accounts, current assets over current liabilities", "controlled_risk.parquet")
        c.add("risk_filiera_structure_2025", f25["structure_index"], f"{f25['structure_index']:.2f}", "01", "2025 accounts, financial-structure index", "controlled_risk.parquet")
        u23 = risk[(risk["company"] == "Unica Reti") & (risk["year"] == 2023)].iloc[0]
        c.add("risk_unica_liquidity_2023", u23["liquidity_index"], f"{u23['liquidity_index']:.2f}", "01", "2023 accounts, current assets over current liabilities", "controlled_risk.parquet")
        ratio_alarms = risk[["structure_alarm", "liquidity_alarm", "charges_alarm"]].any(axis=1)
        c.add("risk_ratio_alarm_years", ratio_alarms.sum(), f"{ratio_alarms.sum()}", "01", "company-years with a ratio below or above its threshold", "controlled_risk.parquet")
    acc = fiscal[(fiscal["regime"] == "dlgs118") & (fiscal["budget_type"] == "consuntivo") & fiscal["year"].isin([2017, 2018])][["comune_id", "year", "rigidity_official_pct"]]
    both = cp[cp["year"].isin([2017, 2018])][["comune_id", "year", "rigidity_cash_pct"]].merge(acc, on=["comune_id", "year"]).dropna()
    corr = float(both["rigidity_cash_pct"].corr(both["rigidity_official_pct"]))
    c.add("rigidity_cash_anchor_corr", corr, f"{corr:.2f}", "01", "2017-2018", "siope_cash_panel.parquet + fiscal.parquet")
    c.add("rigidity_cash_anchor_n", len(both), f"{len(both)}", "01", "2017-2018", "siope_cash_panel.parquet + fiscal.parquet")

    # OpenCivitas
    oc = pd.read_parquet(p / "opencivitas_municipal_summary.parquet")
    pe = oc[oc["is_peer"]].set_index("comune_id")
    src = "opencivitas_municipal_summary.parquet"
    c.add("oc_spending", pe.loc[CESENA, "tot_spesa_storica_eur_pc"], eur(pe.loc[CESENA, "tot_spesa_storica_eur_pc"], 0), "01", "OpenCivitas 2023", src)
    c.add("oc_need", pe.loc[CESENA, "tot_fabbisogno_standard_eur_pc"], eur(pe.loc[CESENA, "tot_fabbisogno_standard_eur_pc"], 0), "01", "OpenCivitas 2023", src)
    c.add("oc_gap", pe.loc[CESENA, "tot_diff_pct"], pct(pe.loc[CESENA, "tot_diff_pct"], 1), "01", "OpenCivitas 2023", src)
    top = int((pe["tot_livello_servizi_0_10"] == 10).sum())
    c.add("oc_top_decile_peers", top, f"{top} of {len(pe)}", "01", "OpenCivitas 2023", src)
    virt = int((pe["quadrante_performance"] == "virtuoso").sum())
    c.add("oc_low_spend_peers", virt, f"{virt} of {len(pe)}", "01", "OpenCivitas 2023", src)
    rk, n = _rank(pe["copertura_nidi_pct"])
    c.add("childcare", pe.loc[CESENA, "copertura_nidi_pct"], share(pe.loc[CESENA, "copertura_nidi_pct"]), "01", "OpenCivitas 2023", src, rk, n)
    c.add("deprivation", pe.loc[CESENA, "deprivazione_indice"], f"{pe.loc[CESENA, 'deprivazione_indice']:.1f}", "01", "OpenCivitas 2023", src)
    below = int((oc[oc["is_belt"]]["deprivazione_indice"] < pe.loc[CESENA, "deprivazione_indice"]).sum())
    c.add("barcelona_2002", 33.0, "33%", "01", "Barcelona target of 2002", "EU Council conclusions, external")
    c.add("barcelona_2030", 45.0, "45%", "01", "Council Recommendation of 2022, for 2030", "EU Council recommendation, external")
    c.add("deprivation_belt_lower", below, f"{below} of {int(oc['is_belt'].sum())}", "01", "OpenCivitas 2023", src)


def _investments(c: _Claims, p: Path) -> None:
    demo = pd.read_parquet(p / "demography.parquet")
    pj = pd.read_parquet(p / "projects.parquet")
    s = summarize_investments_by_comune(pj, demo).set_index("comune_id")
    src = "projects.parquet"
    rk, n = _rank(s["allocated_real_pc"])
    c.add("oc1420_alloc_pc", s.loc[CESENA, "allocated_real_pc"], eur(s.loc[CESENA, "allocated_real_pc"], 0), "02", "2014-2020 cycle", src, rk, n)
    c.add("oc1420_alloc_pc_core", s.loc[CORE, "allocated_real_pc"].median(), eur(s.loc[CORE, "allocated_real_pc"].median(), 0), "02", "2014-2020 cycle", src)
    rk, n = _rank(s["disbursement_rate_pct"])
    c.add("oc1420_disb_rate", s.loc[CESENA, "disbursement_rate_pct"], share(s.loc[CESENA, "disbursement_rate_pct"]), "02", "2014-2020 cycle", src, rk, n)
    c.add("oc1420_disb_rate_core", s.loc[CORE, "disbursement_rate_pct"].median(), share(s.loc[CORE, "disbursement_rate_pct"].median()), "02", "2014-2020 cycle", src)
    c.add("oc1420_pw_n", s.loc[CESENA, "n_public_works"], f"{int(s.loc[CESENA, 'n_public_works'])}", "02", "2014-2020 cycle", src)
    rk, n = _rank(s["public_works_disb_rate_pct"])
    c.add("oc1420_pw_rate", s.loc[CESENA, "public_works_disb_rate_pct"], share(s.loc[CESENA, "public_works_disb_rate_pct"]), "02", "2014-2020 cycle", src, rk, n)
    c.add("oc1420_pw_alloc_pc", s.loc[CESENA, "public_works_alloc_real_pc"], eur(s.loc[CESENA, "public_works_alloc_real_pc"], 0), "02", "2014-2020 cycle", src)
    c.add("oc1420_pw_alloc_pc_core", s.loc[CORE, "public_works_alloc_real_pc"].median(), eur(s.loc[CORE, "public_works_alloc_real_pc"].median(), 0), "02", "2014-2020 cycle", src)

    j = pd.read_parquet(p / "projects_2021_2027.parquet")
    src = "projects_2021_2027.parquet"
    pop = demo[demo["year"] == 2024].set_index("comune_id")["pop_total"]
    tot = j.groupby("comune_id")["allocated_eur"].sum()
    pc = tot / pop.reindex(tot.index)
    rk, n = _rank(pc)
    c.add("oc2127_total", tot[CESENA], meur(tot[CESENA]), "02", "2021-2027, open cycle, nominal", src)
    c.add("oc2127_pc", pc[CESENA], eur(pc[CESENA], 0), "02", "2021-2027, open cycle, nominal", src, rk, n)
    c.add("oc2127_pc_core", pc[CORE].median(), eur(pc[CORE].median(), 0), "02", "2021-2027, open cycle, nominal", src)
    lead = pc.sort_values(ascending=False)
    c.add("oc2127_pc_first", lead.iloc[0], eur(lead.iloc[0], 0), "02", "2021-2027 (Faenza)", src)
    c.add("oc2127_pc_second", lead.iloc[1], eur(lead.iloc[1], 0), "02", "2021-2027 (Bologna)", src)
    growth = (tot[CESENA] / pj[pj["comune_id"] == CESENA]["allocated_eur"].sum() - 1) * 100
    c.add("oc2127_growth", growth, pct(growth, 1), "02", "2021-2027 vs 2014-2020, nominal", src)
    ces = j[j["comune_id"] == CESENA]
    top3 = ces["allocated_eur"].nlargest(3).sum() / ces["allocated_eur"].sum() * 100
    c.add("oc2127_top3", top3, share(top3, 0), "02", "2021-2027", src)
    for k, v in enumerate(ces["allocated_eur"].nlargest(4), 1):
        c.add(f"oc2127_top{k}_eur", v, meur(v), "02", f"2021-2027, project number {k} by amount", src)
    for tag, d, cycle in (("oc1420", pj, "2014-2020"), ("oc2127", j, "2021-2027")):  # located in Cesena is not run by the Comune
        loc = d[d["comune_id"] == CESENA]
        own = loc.loc[loc["is_comune_actor"].astype(bool), "allocated_eur"].sum() / loc["allocated_eur"].sum() * 100
        c.add(f"{tag}_comune_share", own, share(own), "02", f"{cycle}, amount with a Comune as implementing body or beneficiary (name match)", f"projects{'_2021_2027' if tag == 'oc2127' else ''}.parquet")
    disb = ces["disbursed_eur"].sum() / ces["allocated_eur"].sum() * 100
    c.add("oc2127_disb", disb, share(disb, 0), "02", "2021-2027", src)


def _economy(c: _Claims, p: Path) -> None:
    es = pd.read_parquet(p / "employment_sensitivity.parquet")
    src = "employment_sensitivity.parquet"
    fmt = {"employment_growth_pct": pct, "competitive_shift_pp": lambda v: f"{v:+.2f}", "unit_size_change": lambda v: f"{v:+.2f}"}
    short = {"employment_growth_pct": "emp", "competitive_shift_pp": "cs", "unit_size_change": "size"}
    var = {"all_sectors": "all", "without_food_10": "x10", "without_food_agency_cleaning": "x3"}
    for r in es.to_dict("records"):
        text = fmt[str(r["metric"])](float(r["cesena"]))
        c.add(f"{short[str(r['metric'])]}_{var[str(r['variant'])]}_{r['t1']}", float(r["cesena"]), text, "03", f"{r['t0']}-{r['t1']}", src, int(r["rank"]), int(r["n"]))

    a = pd.read_parquet(p / "asia_sectors.parquet")
    src = "asia_sectors.parquet"
    ces = a[(a["comune_id"] == CESENA) & (~a["is_total"].astype(bool))]
    food = ces[ces["ateco_2d"] == "10"].set_index("year")["employees"]
    for y in (2021, 2022, 2023, 2024):
        c.add(f"food_{y}", food[y], num(food[y], 0), "03", str(y), src)
    tot = ces.groupby("year")["employees"].sum()
    net = tot[2024] - tot[2012]
    c.add("net_jobs", net, signed(net, 0), "03", "2012-2024", src)
    for code, name in (("10", "food"), ("78", "agency")):
        s = ces[ces["ateco_2d"] == code].set_index("year")["employees"]
        d = s[2024] - s[2012]
        c.add(f"{name}_jobs", d, signed(d, 0), "03", "2012-2024", src)
        c.add(f"{name}_jobs_share", d / net * 100, share(d / net * 100, 0), "03", "2012-2024", src)
    ag78, cl81 = ces[ces["ateco_2d"] == "78"].set_index("year")["employees"], ces[ces["ateco_2d"] == "81"].set_index("year")["employees"]
    c.add("agency_jobs_2023", ag78[2023] - ag78[2012], signed(ag78[2023] - ag78[2012], 0), "03", "2012-2023", src)
    c.add("agency_change_2024", ag78[2024] - ag78[2023], signed(ag78[2024] - ag78[2023], 0), "03", "2023-2024", src)
    c.add("cleaning_change_2024", cl81[2024] - cl81[2023], signed(cl81[2024] - cl81[2023], 0), "03", "2023-2024", src)
    an = pd.read_parquet(p / "asia_anomalies.parquet")
    c.add("anomalies_n", len(an), f"{len(an)}", "03", "2013-2024", "asia_anomalies.parquet")

    sz = pd.read_parquet(p / "asia_size_classes.parquet")
    med = sz[(sz["comune_id"] == CESENA) & (sz["size_class"] == "W50_249") & (~sz["is_total"].astype(bool))]
    wide = med.pivot_table(index="ateco_2d", columns="year", values="employees", aggfunc="sum").fillna(0.0)
    d_med = float((wide[2024] - wide[2012]).sum())
    c.add("medium_jobs", d_med, signed(d_med, 0), "03", "2012-2024", "asia_size_classes.parquet")
    c.add("medium_agency_share", (wide.loc["78", 2024] - wide.loc["78", 2012]) / d_med * 100, share((wide.loc["78", 2024] - wide.loc["78", 2012]) / d_med * 100, 0), "03", "2012-2024", "asia_size_classes.parquet")

    # R1: the food break against the provinces, and where in Cesena's size classes it sits
    ar = pd.read_parquet(p / "asia_areas.parquet")
    src = "asia_areas.parquet"
    pf = ar[ar["ateco_2d"] == "10"].pivot(index="nuts", columns="year", values="employees")
    for y in (2021, 2022, 2023):
        c.add(f"prov_food_{y}", pf.loc["ITH58", y], num(pf.loc["ITH58", y], 0), "03", str(y), src)
    er_prov = [f"ITH5{k}" for k in range(1, 10)]
    fall = (pf.loc[er_prov, 2022] / pf.loc[er_prov, 2021] - 1) * 100
    c.add("prov_food_fall_n", int((fall < -10).sum()), f"{int((fall < -10).sum())} of {len(er_prov)}", "03", "2021-2022", src)
    er_chg = (pf.loc["ITD5", 2022] / pf.loc["ITD5", 2021] - 1) * 100  # the region keeps its NUTS 2010 code
    c.add("er_food_2022", er_chg, pct(er_chg, 1), "03", "2021-2022", src)
    cesena_drop = food[2021] - food[2022]
    prov_drop = pf.loc["ITH58", 2021] - pf.loc["ITH58", 2022]
    c.add("cesena_share_prov_drop", cesena_drop / prov_drop * 100, share(cesena_drop / prov_drop * 100, 0), "03", "2021-2022", src)
    sw = sector_size_swing(pd.read_parquet(p / "asia_size_classes.parquet"), ("10",))
    big = sw.set_index("size_class").loc["W_GE250"]
    src = "asia_size_classes.parquet"
    for y in (2021, 2022, 2023):
        c.add(f"big_food_{y}", big[f"emp_{y}"], num(big[f"emp_{y}"], 0), "03", str(y), src)
        c.add(f"big_food_lu_{y}", big[f"lu_{y}"], num(big[f"lu_{y}"], 0), "03", str(y), src)
    rest22 = sw[sw["size_class"] != "W_GE250"]["delta_2022"].sum()
    c.add("rest_food_2022", rest22, signed(rest22, 0), "03", "2021-2022", src)
    ac = sector_size_swing(pd.read_parquet(p / "asia_size_classes.parquet"), ("78", "81"), (2023, 2024))
    for y in (2023, 2024):
        c.add(f"agcl_{y}", ac[f"emp_{y}"].sum(), num(ac[f"emp_{y}"].sum(), 0), "03", str(y), src)
    big_ac = ac.set_index("size_class").loc["W_GE250", "delta_2024"]
    c.add("agcl_big_2024", big_ac, signed(big_ac, 0), "03", "2023-2024", src)
    c.add("agcl_rest_2024", ac["delta_2024"].sum() - big_ac, signed(ac["delta_2024"].sum() - big_ac, 0), "03", "2023-2024", src)

    fr = pd.read_parquet(p / "frame_sbs_municipal.parquet")
    fr = fr[fr["is_er_benchmark"].astype(bool)]
    src = "frame_sbs_municipal.parquet"
    for sec, tag in (("totale", "total"), ("industria", "ind")):
        w = fr[fr["macro_sector"] == sec]
        add = w.pivot(index="comune_id", columns="year", values="addetti")
        va = w.pivot(index="comune_id", columns="year", values="valore_aggiunto_real_eur")
        for y in (2022, 2023):
            g = (add[y] / add[2017] - 1) * 100
            rk, n = _rank(g)
            c.add(f"frame_{tag}_jobs_{y}", g[CESENA], pct(g[CESENA]), "03", f"2017-{y}", src, rk, n)
            v = (va[y] / va[2017] - 1) * 100
            rk, n = _rank(v)
            c.add(f"frame_{tag}_va_{y}", v[CESENA], pct(v[CESENA]), "03", f"2017-{y}", src, rk, n)
            pr = ((va[y] / add[y]) / (va[2017] / add[2017]) - 1) * 100
            rk, n = _rank(pr)
            c.add(f"frame_{tag}_prod_{y}", pr[CESENA], pct(pr[CESENA]), "03", f"2017-{y}", src, rk, n)
    y23 = fr[(fr["year"] == 2023) & (fr["macro_sector"] == "totale")].set_index("comune_id")
    c.add("frame_va_worker_2023", y23.loc[CESENA, "va_per_addetto_eur"], eur(y23.loc[CESENA, "va_per_addetto_eur"], 0), "03", "2023, nominal", src)
    c.add("frame_va_worker_2023_core", y23.loc[CORE, "va_per_addetto_eur"].median(), eur(y23.loc[CORE, "va_per_addetto_eur"].median(), 0), "03", "2023, nominal", src)
    c.add("frame_wage_2023", y23.loc[CESENA, "retribuzione_per_dipendente_eur"], eur(y23.loc[CESENA, "retribuzione_per_dipendente_eur"], 0), "03", "2023, nominal", src)
    c.add("frame_wage_2023_core", y23.loc[CORE, "retribuzione_per_dipendente_eur"].median(), eur(y23.loc[CORE, "retribuzione_per_dipendente_eur"].median(), 0), "03", "2023, nominal", src)

    ag = pd.read_parquet(p / "agri_sat_land_use_provinces.parquet").set_index("category")
    src = "agri_sat_land_use_provinces.parquet"
    prov = [x for x in ag.columns if x not in ("is_subitem", "parent_category", "Emilia-Romagna")]
    fruit = ag.loc["Frutta", prov].astype(float)
    rk, n = int((fruit > fruit["Forlì-Cesena"]).sum() + 1), len(fruit)
    c.add("fruit_ha", fruit["Forlì-Cesena"], f"{fruit['Forlì-Cesena']:,.0f} ha", "03", "2020 census", src, rk, n)
    sau = pd.read_parquet(p / "agri_sau_size_classes_provinces.parquet").set_index("size_class")
    fc_sau = float(sau.loc["Totale", "Forlì-Cesena"])
    c.add("fruit_share_fc", fruit["Forlì-Cesena"] / fc_sau * 100, share(fruit["Forlì-Cesena"] / fc_sau * 100), "03", "2020 census", src)
    er_share = float(ag.loc["Frutta", "Emilia-Romagna"]) / float(sau.loc["Totale", "Emilia-Romagna"]) * 100
    c.add("fruit_share_er", er_share, share(er_share), "03", "2020 census", src)
    small = float(sau.loc["< 10 ha", "Forlì-Cesena"]) / fc_sau * 100
    c.add("small_farms_fc", small, share(small), "03", "2020 census", "agri_sau_size_classes_provinces.parquet")
    ex = pd.read_parquet(p / "coeweb_provinces_export.parquet").set_index("provincia").loc["Forlì-Cesena"]
    c.add("exports_2025", ex["export_eur"], beur(ex["export_eur"]), "03", "2025", "coeweb_provinces_export.parquet", int(ex["rank_export"]), 9)
    c.add("trade_balance_2025", ex["trade_balance_eur"], beur(ex["trade_balance_eur"]), "03", "2025", "coeweb_provinces_export.parquet", int(ex["rank_balance"]), 9)


def _outcomes(c: _Claims, p: Path) -> None:
    ic = pd.read_parquet(p / "income_composition.parquet")
    src = "income_composition.parquet"
    short = {"mean_income": "mean", "median_income": "median", "taxpayers": "taxpayers", "aggregate_income": "aggregate", "income_per_resident": "per_resident"}
    for r in ic.to_dict("records"):
        cid = f"income_{short[str(r['measure'])]}_{str(r['t0'])[2:]}{str(r['t1'])[2:]}"
        window = f"{r['t0']}-{r['t1']}"
        c.add(cid, float(r["cesena"]), pct(float(r["cesena"])), "04", window, src, int(r["rank"]), int(r["n"]))
        c.add(f"{cid}_peer", float(r["peer_median"]), pct(float(r["peer_median"])), "04", window, src)

    if REVIEW_GROWTH.exists():  # direct annual HICP index levels instead of the chained annual rates
        rg = pd.read_csv(REVIEW_GROWTH, dtype={"comune_id": str})
        r = rg[(rg["measure"] == "mean_real_direct_hicp_eur") & (rg["comune_id"] == CESENA) & (rg["start"] == 2008) & (rg["end"] == 2024)].iloc[0]
        c.add("income_mean_0824_direct", r["growth_pct"], pct(r["growth_pct"]), "04", "2008-2024, direct annual HICP index", "reports/income_evidence/review_growth.csv", int(r["rank"]), int(r["cities"]))

    co = pd.read_parquet(p / "cohort_by_period.parquet")
    for r in co.to_dict("records"):
        cid = f"cohort_{str(r['t0'])[2:]}{str(r['t1'])[2:]}"
        window = f"{r['t0']}-{r['t1']}"
        c.add(cid, float(r["cesena"]), f"{float(r['cesena']):+.1f}", "04", window, "cohort_by_period.parquet", int(r["rank"]), int(r["n"]))
        c.add(f"{cid}_peer", float(r["peer_median"]), f"{float(r['peer_median']):+.1f}", "04", window, "cohort_by_period.parquet")

    pb = pd.read_parquet(p / "population_balance.parquet").set_index(["area", "t0", "t1"])
    src = "population_balance.parquet"
    for area in ("cesena", "belt_16"):
        r = pb.loc[(area, 2008, 2024)]
        c.add(f"pop_change_{area}", r["pop_change"], signed(r["pop_change"], 0), "04", "2008-2024", src)
        for col, tag in (("pop_change_per_1k", "total"), ("natural_balance_per_1k", "natural"), ("registry_adjustment_per_1k", "adjustment")):
            c.add(f"balance_{tag}_{area}", r[col], signed(r[col]), "04", "2008-2024", src)
        mig = r["net_migration_internal_per_1k"] + r["net_migration_international_per_1k"]
        c.add(f"balance_migration_{area}", mig, signed(mig), "04", "2008-2024", src)
        for t0, t1 in ((2008, 2011), (2012, 2018), (2019, 2024)):
            v = pb.loc[(area, t0, t1), "net_migration_internal"]
            c.add(f"internal_{area}_{str(t0)[2:]}{str(t1)[2:]}", v, signed(v, 0), "04", f"{t0}-{t1}", src)
    a, b = pb.loc[("cesena", 2008, 2024)], pb.loc[("belt_16", 2008, 2024)]
    gap = b["pop_change_per_1k"] - a["pop_change_per_1k"]
    nat = (b["natural_balance_per_1k"] - a["natural_balance_per_1k"]) / gap * 100
    c.add("balance_gap", gap, f"{gap:.1f}", "04", "2008-2024", src)
    c.add("balance_gap_natural", nat, share(nat, 0), "04", "2008-2024", src)
    mig_gap = ((b["net_migration_internal_per_1k"] + b["net_migration_international_per_1k"]) - (a["net_migration_internal_per_1k"] + a["net_migration_international_per_1k"])) / gap * 100
    c.add("balance_gap_migration", mig_gap, share(mig_gap, 0), "04", "2008-2024", src)
    early = pb.loc[("belt_16", 2008, 2011), "pop_change"] / b["pop_change"] * 100
    c.add("belt_growth_0811_share", early, share(early, 0), "04", "2008-2011 of 2008-2024", src)
    v = pb.loc[("belt_16", 2012, 2018), "pop_change"] + pb.loc[("belt_16", 2019, 2024), "pop_change"]
    c.add("belt_growth_1224", v, signed(v, 0), "04", "2012-2024", src)

    pl = pd.read_parquet(p / "placebo_ranking.parquet").set_index("metric")
    src = "placebo_ranking.parquet"
    r = pl.loc["pop_total"]
    c.add("pop_growth", r["cesena_value"], pct(r["cesena_value"]), "04", r["period"], src, int(r["rank_1_to_13"]), int(r["n_comuni"]))
    c.add("pop_growth_peer", r["peer_median_ex_cesena"], pct(r["peer_median_ex_cesena"]), "04", r["period"], src)
    r = pl.loc["aging_index"]
    c.add("aging", r["cesena_value"], signed(r["cesena_value"]), "04", r["period"], src, int(r["rank_1_to_13"]), int(r["n_comuni"]))
    c.add("aging_peer", r["peer_median_ex_cesena"], signed(r["peer_median_ex_cesena"]), "04", r["period"], src)

    # University
    us = pd.read_parquet(p / "mur_university_summary.parquet")
    src = "mur_university_summary.parquet"
    y = us[us["calendar_year"] == 2025].set_index("comune_id")
    pop = pd.read_parquet(p / "demography.parquet")
    pop = pop[pop["year"] == 2025].set_index("comune_id")["pop_total"]
    dens = y["students_total"] / pop.reindex(y.index) * 1000
    rk, n = _rank(dens)
    c.add("students_density", dens[CESENA], f"{dens[CESENA]:.1f}", "04", "2025/26", src, rk, n)
    c.add("students_density_core", dens[CORE].median(), f"{dens[CORE].median():.1f}", "04", "2025/26", src)
    c.add("students_density_all", dens.drop(index=CESENA).median(), f"{dens.drop(index=CESENA).median():.1f}", "04", "2025/26", src)
    c.add("students_total", y.loc[CESENA, "students_total"], num(y.loc[CESENA, "students_total"], 0), "04", "2025/26", src)
    ces = us[us["comune_id"] == CESENA].set_index("calendar_year")["students_total"]
    c.add("students_2010", ces[2010], num(ces[2010], 0), "04", "2010/11", src)
    c.add("students_2019", ces[2019], num(ces[2019], 0), "04", "2019/20", src)
    for col, tag in (("ict_share", "ict"), ("engineering_share", "engineering"), ("stem_share", "stem")):
        big = y[y["students_total"] >= 1000]
        rk, n = _rank(big[col])
        c.add(f"uni_{tag}", y.loc[CESENA, col] * 100, share(y.loc[CESENA, col] * 100), "04", "2025/26", src, rk, n)
    c.add("uni_ict_students", y.loc[CESENA, "students_ict"], num(y.loc[CESENA, "students_ict"], 0), "04", "2025/26", src)
    c.add("uni_grads", y.loc[CESENA, "graduates_total"], num(y.loc[CESENA, "graduates_total"], 0), "04", "2025", src)

    # Housing
    om = pd.read_parquet(p / "omi_quotations_municipal.parquet")
    pe = om[om["is_er_peer"] | om["is_cesena"]].set_index("comune_id")
    src = "omi_quotations_municipal.parquet"
    rk, n = _rank(pe["compr_mid_eur_m2"])
    c.add("omi_price", pe.loc[CESENA, "compr_mid_eur_m2"], eur(pe.loc[CESENA, "compr_mid_eur_m2"], 0), "04", "OMI quotations", src, rk, n)
    rk, n = _rank(pe["loc_mid_eur_m2_mo"])
    c.add("omi_rent", pe.loc[CESENA, "loc_mid_eur_m2_mo"], eur(pe.loc[CESENA, "loc_mid_eur_m2_mo"], 2), "04", "OMI quotations", src, rk, n)
    grad = om[~om["is_er_peer"] & ~om["is_cesena"]].groupby("belt_group")["diff_vs_cesena_pct"].median()
    for g, tag in (("rubicone_mare", "rubicone"), ("via_emilia_west", "emilia"), ("unione_valle_savio", "savio")):
        c.add(f"omi_discount_{tag}", grad[g], pct(grad[g], 1), "04", "OMI quotations, median town", src)
    ht = pd.read_parquet(p / "housing_gradient_test.parquet").set_index("y")
    for key, tag in (("pop_growth_2008_2025", "belt_0825"), ("pop_growth_2012_2024", "belt_1224"), ("cohort_25_44_2012_2024", "cities_1224")):
        c.add(f"gradient_{tag}", ht.loc[key, "spearman"], f"{ht.loc[key, 'spearman']:+.2f}", "04", key, "housing_gradient_test.parquet")
    belt = pd.read_parquet(p / "belt_comuni.parquet")
    b24 = belt[belt["year"] == 2024].copy()
    b24["rate"] = b24["addizionale_comunale_real_eur"] / b24["income_pos_real_eur"] * 100.0
    ces_rate = float(b24[b24["comune_id"] == CESENA]["rate"].iloc[0])
    others = b24[b24["comune_id"] != CESENA]
    c.add("surcharge_rate_cesena", ces_rate, share(ces_rate, 2), "04", "2024", "belt_comuni.parquet")
    belt_rate = float(others["addizionale_comunale_real_eur"].sum() / others["income_pos_real_eur"].sum() * 100.0)
    c.add("surcharge_rate_belt", belt_rate, share(belt_rate, 2), "04", "2024", "belt_comuni.parquet")
    lower = int((others["rate"].round(2) <= round(ces_rate, 2)).sum())
    c.add("surcharge_towns_not_higher", lower, f"{lower} of {len(others)}", "04", "2024", "belt_comuni.parquet")

    # Commuting (2011 census)
    cm = pd.read_parquet(p / "commuting_municipal_summary.parquet")
    pe = cm[cm["is_peer"]].set_index("comune_id")
    src = "commuting_municipal_summary.parquet"
    rk, n = _rank(pe["net_commuter_inflow"])
    c.add("commute_net", pe.loc[CESENA, "net_commuter_inflow"], signed(pe.loc[CESENA, "net_commuter_inflow"], 0), "04", "2011 census", src, rk, n)
    rk, n = _rank(pe["job_self_containment_rate"])
    c.add("commute_self", pe.loc[CESENA, "job_self_containment_rate"] * 100, share(pe.loc[CESENA, "job_self_containment_rate"] * 100), "04", "2011 census", src, rk, n)
    co2 = pd.read_parquet(p / "commuting_belt_corridors.parquet")
    belt = co2[co2["group"].isin(["rubicone_mare", "via_emilia_west", "unione_valle_savio"])]
    inb = belt["inbound_to_cesena"].sum()
    c.add("commute_belt_in", inb, num(inb, 0), "04", "2011 census", "commuting_belt_corridors.parquet")
    c.add("commute_belt_share", inb / pe.loc[CESENA, "inbound_workers"] * 100, share(inb / pe.loc[CESENA, "inbound_workers"] * 100, 0), "04", "2011 census", src)
    stud = belt["study_inbound_to_cesena"].sum() / pe.loc[CESENA, "inbound_students"] * 100
    c.add("commute_students_belt", stud, share(stud, 0), "04", "2011 census", src)


def _evaluation(c: _Claims, p: Path) -> None:
    ex = pd.read_parquet(p / "extended_placebo_ranking.parquet").set_index(["pool_key", "metric"])
    src = "extended_placebo_ranking.parquet"
    for pool, tag in (("NC_midsize_40k_120k", "nc45"), ("IT_midsize_40k_120k", "it83"), ("ER_15k_payers", "er36"), ("ER_10k_payers", "er73")):
        for metric, m in (("mean_income_growth_08_24_pct", "mean"), ("median_income_growth_08_24_pct", "median")):
            r = ex.loc[(pool, metric)]
            c.add(f"pool_{tag}_{m}", r["cesena_value"], pct(r["cesena_value"]), "05", "2008-2024", src, int(r["rank"]), int(r["n_comuni"]))
            c.add(f"pool_{tag}_{m}_peer", r["peer_median_ex_cesena"], pct(r["peer_median_ex_cesena"]), "05", "2008-2024", src)
    q = float(ex["q_bh"].min())
    c.add("pool_min_q", q, f"{q:.2f}", "05", "16 rank proportions", src)
    sens = pd.read_parquet(p / "extended_pool_sensitivity.parquet")
    sm = sens[(sens["metric"] == "mean_income_growth_08_24_pct") & (sens["region"] == "Italy")].set_index("taxpayers_lo")
    for lo in (10000, 20000, 50000):
        c.add(f"pool_italy_{lo // 1000}k", sm.loc[lo, "rank"], f"#{int(sm.loc[lo, 'rank'])} of {int(sm.loc[lo, 'n'])}", "05", f"taxpayers from {lo}", "extended_pool_sensitivity.parquet")

    sp = pd.read_parquet(p / "synthetic_in_space.parquet")
    sp = sp[sp["is_cesena"]].set_index("metric")
    tag = {"employees": "emp", "avg_firm_size": "size", "mean_income_real_eur": "mean", "median_income_real_eur": "median"}
    for metric, t in tag.items():
        r = sp.loc[metric]
        c.add(f"sc_{t}_gap", r["gap_end_pp"], pp(r["gap_end_pp"]), "05", "2024", "synthetic_in_space.parquet", int(r["rank_gap_desc"]), int(r["n_units"]))
        c.add(f"sc_{t}_ratio", r["ratio_post_pre_rmspe"], f"{r['ratio_post_pre_rmspe']:.1f}", "05", "2019-2024 vs before", "synthetic_in_space.parquet", int(r["rank_ratio_desc"]), int(r["n_units"]))
    it = pd.read_parquet(p / "synthetic_in_time.parquet").set_index("metric")
    c.add("sc_mean_intime", it.loc["mean_income_real_eur", "pseudo_post_mean_gap_pp"], pp(it.loc["mean_income_real_eur", "pseudo_post_mean_gap_pp"]), "05", "2015-2018, split at 2014", "synthetic_in_time.parquet")
    s = pd.read_parquet(p / "synthetic_cesena.parquet")
    g18 = s[(s["metric"] == "mean_income_real_eur") & (s["year"] == 2018)]["gap_vs_synthetic_pp"].iloc[0]
    c.add("sc_mean_gap_2018", g18, pp(g18), "05", "2018", "synthetic_cesena.parquet")
    lo = pd.read_parquet(p / "synthetic_leave_one_out.parquet")
    v = lo[(lo["metric"] == "mean_income_real_eur") & (lo["dropped"] == "Ferrara")]["gap_end_pp"].iloc[0]
    c.add("sc_mean_no_ferrara", v, pp(v), "05", "2024", "synthetic_leave_one_out.parquet")
    w = lo[(lo["metric"] == "mean_income_real_eur") & (lo["dropped"] == "Ferrara")]["weight_dropped"].iloc[0]
    c.add("sc_mean_ferrara_weight", w * 100, share(w * 100, 0), "05", "2008-2018 fit", "synthetic_leave_one_out.parquet")

    med = lo[lo["metric"] == "median_income_real_eur"].set_index("dropped")
    c.add("sc_median_faenza_weight", med.loc["Faenza", "weight_dropped"] * 100, share(med.loc["Faenza", "weight_dropped"] * 100, 0), "05", "2008-2018 fit", "synthetic_leave_one_out.parquet")
    c.add("sc_median_no_faenza", med.loc["Faenza", "gap_end_pp"], pp(med.loc["Faenza", "gap_end_pp"]), "05", "2024", "synthetic_leave_one_out.parquet")
    c.add("sc_median_no_faenza_prefit", med.loc["Faenza", "pre_rmse_pp"], f"{med.loc['Faenza', 'pre_rmse_pp']:.1f} pp", "05", "2008-2018 fit error without Faenza", "synthetic_leave_one_out.parquet")
    c.add("sc_median_prefit", med.loc["none", "pre_rmse_pp"], f"{med.loc['none', 'pre_rmse_pp']:.1f} pp", "05", "2008-2018 fit error", "synthetic_leave_one_out.parquet")

    lf = pd.read_parquet(p / "lagged_fe_results.parquet")
    n = int(lf["p_perm"].notna().sum())
    c.add("fe_tests", n, f"{n}", "05", "lags 1-4", "lagged_fe_results.parquet")
    c.add("fe_min_q", lf["q_bh_perm"].min(), f"{lf['q_bh_perm'].min():.2f}", "05", "lags 1-4", "lagged_fe_results.parquet")
    lr = pd.read_parquet(p / "lagged_fe_regime_results.parquet")
    c.add("fe_regime_min_q", lr["q_bh_perm"].min(), f"{lr['q_bh_perm'].min():.2f}", "05", "lags 1-4, regime FE", "lagged_fe_regime_results.parquet")

    nu = pd.read_parquet(p / "shuffled_null.parquet").iloc[0]
    c.add("fe_null_runs", nu["n_rep"], f"{int(nu['n_rep'])}", "05", "shuffled predictors", "shuffled_null.parquet")
    c.add("fe_null_share", nu["share_runs_with_discovery"] * 100, share(nu["share_runs_with_discovery"] * 100, 0), "05", "shuffled predictors", "shuffled_null.parquet")

    fl = pd.read_parquet(p / "flood_shock_panel.parquet")
    src = "flood_shock_panel.parquet"
    fb = fl.set_index("comune_id")
    for cid, name in ((CESENA, "cesena"), ("039010", "faenza"), ("037032", "imola")):
        c.add(f"flood_income_{name}", fb.loc[cid, "income_growth_22_23_pct"], pct(fb.loc[cid, "income_growth_22_23_pct"]), "05", "2022-2023", src)
    tier = fl.groupby("flood_tier")[["income_growth_22_23_pct", "capitale_cassa_growth_22_24_pct", "capitale_cassa_growth_base_24_pct", "emp_growth_x10_22_24_pct"]].median()
    for t, name in (("whole_territory", "whole"), ("not_listed", "unlisted")):
        c.add(f"flood_income_{name}", tier.loc[t, "income_growth_22_23_pct"], pct(tier.loc[t, "income_growth_22_23_pct"]), "05", "2022-2023", src)
        c.add(f"flood_capex22_{name}", tier.loc[t, "capitale_cassa_growth_22_24_pct"], pct(tier.loc[t, "capitale_cassa_growth_22_24_pct"], 1), "05", "2022-2024", src)
        c.add(f"flood_capexbase_{name}", tier.loc[t, "capitale_cassa_growth_base_24_pct"], pct(tier.loc[t, "capitale_cassa_growth_base_24_pct"], 1), "05", "2019-21 to 2024", src)
    c.add("flood_capex22_cesena", fb.loc[CESENA, "capitale_cassa_growth_22_24_pct"], pct(fb.loc[CESENA, "capitale_cassa_growth_22_24_pct"], 1), "05", "2022-2024", src)
    c.add("flood_capexbase_cesena", fb.loc[CESENA, "capitale_cassa_growth_base_24_pct"], pct(fb.loc[CESENA, "capitale_cassa_growth_base_24_pct"], 1), "05", "2019-21 to 2024", src)
    c.add("flood_emp_all_cesena", fb.loc[CESENA, "emp_growth_22_24_pct"], pct(fb.loc[CESENA, "emp_growth_22_24_pct"]), "05", "2022-2024", src)
    c.add("flood_emp_x10_cesena", fb.loc[CESENA, "emp_growth_x10_22_24_pct"], pct(fb.loc[CESENA, "emp_growth_x10_22_24_pct"]), "05", "2022-2024", src)
    c.add("flood_emp_x10_unlisted", tier.loc["not_listed", "emp_growth_x10_22_24_pct"], pct(tier.loc["not_listed", "emp_growth_x10_22_24_pct"]), "05", "2022-2024", src)


def income_followup_claims(components: pd.DataFrame, deflator: pd.Series) -> pd.DataFrame:
    """Quoted source comparisons for article 01, on an explicitly supplied price basis."""
    values = component_changes(components, deflator)
    values = values[values.comune_id.eq('040007') & values.component.isin(['reddito_dipendente', 'reddito_pensione'])].copy()
    return values


def build_claims(processed_dir: Path = PROCESSED_DIR, write: bool = True) -> pd.DataFrame:
    """Compute all quoted figures; `id` is unique and `text` is the exact string `findings.md` must show."""
    c = _Claims()
    for part in (_fiscal, _investments, _economy, _outcomes, _evaluation):
        part(c, processed_dir)
    out = pd.DataFrame(c.rows)
    if out["id"].duplicated().any():
        raise ValueError(f"Duplicate claim ids: {sorted(out[out['id'].duplicated()]['id'])}")
    if not np.isfinite(out["value"].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite claims: {sorted(out[~np.isfinite(out['value'].astype(float))]['id'])}")
    if write:
        out.to_parquet(processed_dir / "claims.parquet", index=False)
    return out


def claims_for_notebook(notebook: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Claims of one notebook (`"01"`..`"05"`), to display next to its takeaways."""
    d = pd.read_parquet(processed_dir / "claims.parquet")
    return d[d["notebook"] == notebook][["id", "text", "window", "source"]].reset_index(drop=True)


def claim_text(cid: str, processed_dir: Path = PROCESSED_DIR) -> str:
    """Printed text of one claim (for the dashboard cards)."""
    d = pd.read_parquet(processed_dir / "claims.parquet").set_index("id")
    return str(d.loc[cid, "text"])


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 50)
    print(build_claims()[["id", "text", "window"]].to_string())
