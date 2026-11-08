"""Recompute headline Cesena figures straight from the raw files, without the study's analysis modules or the shared loaders, and compare them with `claims.parquet`.

An independent check on the pipeline: if a loader or a table is wrong, this disagrees. Exits non-zero on any mismatch.
Usage: uv run python scripts/recheck_cesena_claims.py
"""
import csv
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
CLAIMS = Path("data/processed/cesena/claims.parquet")
CES_CAT, CES_ID = "C573", "040007"
CITIES = {"C573": "Cesena", "D704": "Forlì", "H294": "Rimini", "H199": "Ravenna", "E289": "Imola", "B819": "Carpi", "H223": "Reggio", "D458": "Faenza", "D548": "Ferrara", "F257": "Modena", "G337": "Parma", "G535": "Piacenza", "A944": "Bologna"}


def price_index() -> dict[int, float]:
    """Italian HICP chained from the panel's yearly inflation, 2020 = 1."""
    p = pd.read_parquet("data/processed/panel.parquet")
    inf = p[(p["iso3"] == "ITA") & (p["indicator"] == "hicp_inflation")].sort_values("year").set_index("year")["value"]
    idx = {int(inf.index.min()) - 1: 1.0}
    for y, v in inf.items():
        idx[int(y)] = idx[int(y) - 1] * (1 + float(v) / 100)
    return {y: v / idx[2020] for y, v in idx.items()}


def mef_rows(year: int) -> dict[str, dict[str, str]]:
    with zipfile.ZipFile(RAW / "mef_comuni" / f"irpef_comuni_{year}.zip") as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        text = z.read(name).decode("latin1").splitlines()
    head = [" ".join(h.split()) for h in text[0].split(";")]
    cat = next(i for i, h in enumerate(head) if "codice catastale" in h.lower())
    return {r[cat].strip().upper(): dict(zip(head, r, strict=False)) for r in csv.reader(text[1:], delimiter=";") if len(r) > cat and r[cat].strip().upper() in CITIES}


def number(s: str | None) -> float:
    return float(s.replace(",", ".")) if s and s.strip() else 0.0


def real_mean(row: dict[str, str], year: int, px: dict[int, float]) -> float:
    """Total over filers from 2023, when the total is published and the top bracket may be blank; sum of positive brackets before."""
    if "Reddito complessivo - Ammontare in euro" in row:
        return number(row["Reddito complessivo - Ammontare in euro"]) / number(row["Reddito complessivo - Frequenza"]) / px[year]
    amt = sum(number(v) for k, v in row.items() if k.startswith("Reddito complessivo ") and "Ammontare" in k and "minore" not in k)
    frq = sum(number(v) for k, v in row.items() if k.startswith("Reddito complessivo ") and "Frequenza" in k and "minore" not in k)
    return amt / frq / px[year]


def asia_employees() -> pd.DataFrame:
    d = pd.read_csv(RAW / "istat_comuni" / "asia_muni_2d.csv", usecols=["REF_AREA", "DATA_TYPE", "ECON_ACTIVITY_NACE_2007", "PERS_EMPL_SIZE_CLASS", "TIME_PERIOD", "OBS_VALUE"], dtype={"ECON_ACTIVITY_NACE_2007": str})
    d = d[(d["DATA_TYPE"] == "LUEMPDAA") & (d["PERS_EMPL_SIZE_CLASS"] == "TOTAL") & d["ECON_ACTIVITY_NACE_2007"].str.fullmatch(r"\d{2}")]
    return d[d["REF_AREA"].astype(str).str.lstrip("0") == CES_ID.lstrip("0")]


def balance() -> dict[str, float]:
    """Cesena's population change 2008-2024 by component, from the ISTAT files."""
    h = pd.read_csv(RAW / "istat_comuni" / "demo_hist_2008_2019.csv", low_memory=False)
    b = pd.read_csv(RAW / "istat_comuni" / "demo_bal_2019_2025.csv", low_memory=False)
    h = h[(h["REF_AREA"].astype(str).str.lstrip("0") == CES_ID.lstrip("0")) & (h["AGE"] == "TOTAL")]
    b = b[b["REF_AREA"].astype(str).str.lstrip("0") == CES_ID.lstrip("0")]
    flow = pd.concat([h[h["TIME_PERIOD"] < 2019], b[b["TIME_PERIOD"] <= 2024]]).pivot_table(index="TIME_PERIOD", columns="DATA_TYPE", values="OBS_VALUE", aggfunc="first")
    natural = (flow["LBIRTH"] - flow["DEATH"]).sum()
    migration = (flow["REGM"] - flow["DEREGM"] + flow["REGOTHC"] - flow["DEREGOTHC"]).sum()
    start = float(h[(h["DATA_TYPE"] == "JAN") & (h["TIME_PERIOD"] == 2008)]["OBS_VALUE"].iloc[0])
    end = float(b[(b["DATA_TYPE"] == "BEG") & (b["TIME_PERIOD"] == 2025)]["OBS_VALUE"].iloc[0])
    return {"natural": natural / start * 1000, "migration": migration / start * 1000, "total": (end - start) / start * 1000}


def consolidated_current_2024() -> float:
    d = pd.read_parquet(RAW / "siope" / "siope_emr_2024_m12.parquet")
    code = d["Codice Gestionale Enti Locali"].astype(str)
    ces = d[d["Descrizione Ente BDAP"] == "COMUNE DI CESENA"]
    uni = d[d["Descrizione Ente BDAP"] == "UNIONE DEI COMUNI VALLE DEL SAVIO"]
    current = lambda t: t[code.loc[t.index].str.startswith("U1")]["Importo cumulato"].sum()
    only = lambda t, c: t[code.loc[t.index] == c]["Importo cumulato"].sum()
    pop = pd.read_csv(RAW / "istat_comuni" / "demo_bal_2019_2025.csv", low_memory=False)
    pop = pop[(pop["REF_AREA"].astype(str).str.lstrip("0") == CES_ID.lstrip("0")) & (pop["DATA_TYPE"] == "BEG") & (pop["TIME_PERIOD"] == 2024)]["OBS_VALUE"].iloc[0]
    return (current(ces) - only(ces, "U1040102005") + 0.834 * (current(uni) - only(uni, "U1040102003"))) / float(pop)


def cash_rigidity_2024() -> float:
    """(personnel U101 + interest U107 + loan repayment U4) / current revenue (E1 + E2 + E3), Cesena 2024, cash, straight from the SIOPE files."""
    sp = pd.read_parquet(RAW / "siope" / "siope_emr_2024_m12.parquet")
    en = pd.read_parquet(RAW / "siope" / "siope_entrate_emr_2024_m12.parquet")
    sp, en = sp[sp["Descrizione Ente BDAP"] == "COMUNE DI CESENA"], en[en["Descrizione Ente BDAP"] == "COMUNE DI CESENA"]
    pay = lambda t, prefixes: t[t["Codice Gestionale Enti Locali"].astype(str).str.startswith(prefixes)]["Importo cumulato"].sum()
    return float((pay(sp, ("U101", "U107", "U4")) / pay(en, ("E1", "E2", "E3"))) * 100)


def accrual_rigidity_2024() -> float:
    """Cesena 2024 accrual ratio straight from the two Rendiconto CSVs (pandas, decimal comma): macroaggregates 1 and 7 of Titolo 1 and the Titolo 4 total over the Titoli 1-3 Tipologie."""
    d = RAW / "trasparenza" / "cesena" / "files" / "4349166"
    en = pd.read_csv(d / "10694601_ENTRATERENDICONTO2024C.csv", sep=";", decimal=",", skiprows=2, encoding="latin1")
    sp = pd.read_csv(d / "10694602_SPESERENDICONTO2024C.csv", sep=";", decimal=",", skiprows=2, encoding="latin1", header=None, names=None).iloc[1:]
    en = en[en["Titolo"].str[:2].isin(["1-", "2-", "3-"]) & en["Tipologia"].fillna("").str.startswith("TIPOLOGIA")]
    t, m, last = sp[0].astype(str), sp[1].fillna("").astype(str), pd.to_numeric(sp[sp.columns[-3]].astype(str).str.replace(",", "."), errors="coerce")
    fixed = last[t.str.startswith("1-") & (m.str.startswith("MACROAGGR. 1 -") | m.str.startswith("MACROAGGR. 7 "))].sum() + last[t.str.startswith("TOTALE TITOLO 4")].sum()
    return float(fixed / en["Accertamenti"].sum() * 100)


def indicator_rigidity_2024() -> float:
    """Indicator 1.1 of the 2024 rendiconto Piano degli indicatori: first line of the row, last token, read with pdftotext outside the project parser."""
    import subprocess
    pdf = RAW / "trasparenza" / "cesena" / "files" / "4416146" / "10805311_Pianoindicatorierisultatiattesi-Rendiconto2024.pdf"
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    line = next(ln for ln in text.splitlines() if ln.startswith("1.1 Incidenza spese rigide"))
    return float(line.split()[-1].replace(",", "."))


def conto_annuale_cost_2024() -> float:
    """Total annual labour cost 2024 (Tab. 12+13+14), last number of its row in the 2024 Conto annuale PDF read with pdftotext."""
    import subprocess
    pdf = RAW / "trasparenza" / "cesena" / "files" / "5665903" / "13550599_contoannuale2024cesena.pdf"
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    line = next(ln for ln in text.splitlines() if "Totale costo annuo del lavoro" in ln)
    return float(line.split()[-1].replace(".", "")) / 1e6


def decentralised_fund_2024() -> float:
    """Resources of the `Fondo risorse decentrate` 2024 (third number of its summary row, Contrattazione Integrativa page) in the 2024 Conto annuale PDF."""
    import subprocess
    pdf = RAW / "trasparenza" / "cesena" / "files" / "5665903" / "13550599_contoannuale2024cesena.pdf"
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    line = next(ln for ln in text.splitlines() if "Fondo risorse decentrate" in ln and "PERSONALE NON DIRIGENTE" in ln)
    return float(line.split("Fondo risorse decentrate")[1].split()[2].replace(".", ""))


def dup_personnel_2025() -> tuple[float, float]:
    """September 2025 DUP, art. 1 c. 557 table: outturn 2024 total (first number of the second `Totale spesa personale` row) and the 2028 plan for staff transferred to the Unione (last number of its row)."""
    import subprocess
    pdf = next((RAW / "trasparenza" / "cesena" / "files").glob("*/11454546_DUP2026_2029*.pdf"))
    lines = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout.splitlines()
    num = lambda t: float(t.replace(".", "").replace(",", "."))
    tot = [ln for ln in lines if "Totale spesa personale" in ln][1]
    unione = next(ln for ln in lines if "Personale trasferito ad Unione" in ln)
    return num(tot.split()[-5]), num(unione.split()[-1])


def filiera_liquidity_2025() -> float:
    """Liquidity index of Filiera Ortofrutticola Romagnola in the 2025 risk report: second company column (index 4 of the tokens) of the current-assets row, read with pdftotext."""
    import subprocess
    pdf = next((RAW / "trasparenza" / "cesena" / "files").glob("*/13617007_Rischiodicrisiaziendale*.pdf"))
    lines = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout.splitlines()
    row = next(ln for ln in lines if ln.strip().startswith("inferiore a 1"))
    return float(row.split()[4].replace(",", "."))


def tari_levy_2022() -> float:
    """Accrued TARI levy 2022 (first amount of the `Domestiche` row of the Scheda Tassa sui Rifiuti) in the 2024 report to the Rendiconto."""
    import subprocess
    pdf = next((RAW / "trasparenza" / "cesena" / "files").glob("*/10694589_All.C1_*.pdf"))
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    line = next(ln for ln in text[text.index("Scheda Tassa sui Rifiuti"):].splitlines() if "Domestiche" in ln)
    return float(line.split("Domestiche")[1].split()[0].replace(".", "").replace(",", "."))


def bologna_rent() -> float:
    rows = json.loads((RAW / "agenzia_entrate_omi" / "bologna_omi.json").read_text())
    res = [r for r in rows if str(r["comune"]).strip().upper() == "BOLOGNA" and ("civili" in str(r["tipo"]).lower() or "economico" in str(r["tipo"]).lower())]
    lo = [float(r["loc_min"]) for r in res if float(r["loc_min"]) > 0]
    hi = [float(r["loc_max"]) for r in res if float(r["loc_max"]) > 0]
    return (sum(lo) / len(lo) + sum(hi) / len(hi)) / 2


def main() -> int:
    px = price_index()
    m08, m22, m23, m24 = mef_rows(2008), mef_rows(2022), mef_rows(2023), mef_rows(2024)
    growth = {k: (real_mean(m24[k], 2024, px) / real_mean(m08[k], 2008, px) - 1) * 100 for k in CITIES}
    emp = asia_employees().groupby(["TIME_PERIOD"])["OBS_VALUE"].sum()
    food = asia_employees()
    food = food[food["ECON_ACTIVITY_NACE_2007"] == "10"].set_index("TIME_PERIOD")["OBS_VALUE"]
    bal = balance()
    pop24 = pd.read_csv(RAW / "istat_comuni" / "demo_bal_2019_2025.csv", low_memory=False)
    pop24 = float(pop24[(pop24["REF_AREA"].astype(str).str.lstrip("0") == CES_ID.lstrip("0")) & (pop24["DATA_TYPE"] == "BEG") & (pop24["TIME_PERIOD"] == 2024)]["OBS_VALUE"].iloc[0])
    checks = {
        "income_mean_0824": (growth["C573"], 0.02),
        "income_mean_0824_rank": (float(sum(v > growth["C573"] for v in growth.values()) + 1), 0.0),
        "flood_income_faenza": ((real_mean(m23["D458"], 2023, px) / real_mean(m22["D458"], 2022, px) - 1) * 100, 0.05),
        "flood_income_imola": ((real_mean(m23["E289"], 2023, px) / real_mean(m22["E289"], 2022, px) - 1) * 100, 0.05),
        "emp_all_2024": ((emp[2024] / emp[2012] - 1) * 100, 0.02),
        "emp_all_2022": ((emp[2022] / emp[2012] - 1) * 100, 0.02),
        "food_2022": (float(food[2022]), 1.0),
        "food_2023": (float(food[2023]), 1.0),
        "balance_natural_cesena": (bal["natural"], 0.1),
        "balance_migration_cesena": (bal["migration"], 0.1),
        "balance_total_cesena": (bal["total"], 0.1),
        "consolidated_current_2024": (consolidated_current_2024(), 0.5),
        "rigidity_cash_2024": (cash_rigidity_2024(), 0.05),
        "rigidity_accrual_2024": (accrual_rigidity_2024(), 0.05),
        "conto_cost_2024": (conto_annuale_cost_2024() * 1e6, 0.5),
        "conto_fund_2024": (decentralised_fund_2024(), 0.5),
        "limit_total_2024": (dup_personnel_2025()[0], 0.5),
        "limit_plan_unione_2028": (dup_personnel_2025()[1], 0.5),
        "risk_filiera_liquidity_2025": (filiera_liquidity_2025(), 0.0005),
        "tari_levy_2022": (tari_levy_2022(), 0.5),
        "rigidity_official_ind_2024": (indicator_rigidity_2024(), 0.0005),
        "surcharge_2024": (number(m24["C573"]["Addizionale comunale dovuta - Ammontare in euro"]) / px[2024] / pop24, 0.5),
    }
    claims = pd.read_parquet(CLAIMS).set_index("id")["value"]
    failed = 0
    print(f"{'claim':32s}{'claims.parquet':>16s}{'from raw':>12s}  ok")
    for cid, (val, tol) in checks.items():
        ok = abs(float(claims[cid]) - val) <= tol + 1e-9
        failed += not ok
        print(f"{cid:32s}{float(claims[cid]):16.3f}{val:12.3f}  {'yes' if ok else 'NO'}")
    rent = pd.read_parquet("data/processed/cesena/omi_quotations_municipal.parquet").set_index("comune_name").loc["Bologna", "loc_mid_eur_m2_mo"]
    ok = abs(float(rent) - bologna_rent()) <= 0.05
    failed += not ok
    print(f"{'Bologna rent (OMI table)':32s}{float(rent):16.3f}{bologna_rent():12.3f}  {'yes' if ok else 'NO'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
