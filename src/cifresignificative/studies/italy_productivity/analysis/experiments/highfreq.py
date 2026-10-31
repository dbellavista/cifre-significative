"""Monthly and quarterly dynamics, event windows and the dated Italian events."""
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from cifresignificative.studies.italy_productivity.analysis import corr
from cifresignificative.studies.italy_productivity.analysis import hf as H

from .common import OUT

HF_CYCLE_PAIRS = [("m_esi", "m_industrial_production"), ("m_industry_confidence", "m_industrial_production"), ("m_consumer_confidence", "m_retail_volume"),
                  ("m_industrial_production", "m_unemployment_rate"), ("m_esi", "m_unemployment_rate"), ("m_services_confidence", "m_unemployment_rate")]
HF_EVENT_SERIES = {"spread_ITA_DEU": "Italy-Germany 10y spread (pp)", "m_long_term_yield": "Italy 10y yield (%)", "m_esi": "Italy economic sentiment", "m_unemployment_rate": "Italy unemployment rate (%)"}


def italian_events(start: str = "2005-01-01") -> pd.DataFrame:
    """Dated events since `start`: Italian government changes, Italian policies (data/external) and ECB decisions."""
    ext = Path("data/external")
    gov = pd.read_csv(ext / "italy_governments.csv", parse_dates=["start"])
    pol = pd.read_csv(ext / "italy_policies.csv", parse_dates=["date"])
    ecb = pd.read_csv(ext / "ecb_events.csv", parse_dates=["date"])
    ev = pd.concat([gov.rename(columns={"government": "event", "start": "date"}).assign(type="government")[["event", "date", "type"]],
                    pol.rename(columns={"title": "event"}).assign(type="policy")[["event", "date", "type"]],
                    ecb.rename(columns={"title": "event"}).assign(type="ecb")[["event", "date", "type"]]], ignore_index=True)
    return ev[ev.date >= start].sort_values("date").reset_index(drop=True)


EVIDENCE_RANK = {"unverified": 0, "partly verified": 1, "verified": 2}


def evidence_levels() -> dict[str, pd.DataFrame]:
    """Rating changes at each distinct evidence level, broadest first: a level that keeps the same rows as the previous one is dropped (when every row is `verified` there is only "all"), so no family repeats the same events."""
    out: dict[str, pd.DataFrame] = {}
    for name, level in (("all", "unverified"), ("partly verified or better", "partly verified"), ("verified", "verified")):
        ra = rating_actions(level)
        if len(ra) and (not out or len(ra) < len(list(out.values())[-1])): out[name] = ra
    return out


def rating_actions(min_evidence: str = "unverified") -> pd.DataFrame:
    """Italy sovereign rating/outlook changes (data/external/rating_actions.csv; changes only, confirmations excluded) as `event, date, type, action, agency`. Kept apart from `italian_events` so the monthly event study is unchanged; `min_evidence` keeps rows at or above that level of the ladder `unverified` < `partly verified` (two sources agree on the day) < `verified` (the day is in the agency's own record)."""
    ra = pd.read_csv(Path("data/external/rating_actions.csv"), parse_dates=["date"], keep_default_na=False)
    ra = ra[ra.evidence.map(EVIDENCE_RANK) >= EVIDENCE_RANK[min_evidence]]
    score = {"negative": -1, "stable": 0, "positive": 1}
    better = np.where(ra.action == "upgrade", 1, np.where(ra.action == "downgrade", -1, ra.outlook_to.map(score) - ra.outlook_from.map(score)))
    ra = ra.assign(event=ra.agency + " " + ra.action + " " + ra.rating_to.where(ra.action != "outlook", ra.outlook_to), type="rating", direction=-np.sign(better).astype(int))  # +1: spread expected to widen
    return ra[["event", "date", "type", "action", "agency", "evidence", "direction"]].sort_values("date").reset_index(drop=True)


def run_hf_stationarity(hf: pd.DataFrame) -> pd.DataFrame:
    extra = [("m_hicp_inflation", "level"), ("q_labour_cost_yoy", "level"), ("m_esi", "diff"), ("m_consumer_confidence", "level")]  # alternative transforms for comparison
    return H.stationarity_summary(H.stationarity_report(hf, [*H.TRANSFORMS.items(), *extra]))


def run_hf_spread(hf: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([H.spread_model(hf, c) for c in ("ITA", "ESP", "FRA", "PRT", "GRC")], ignore_index=True)
    return out.assign(q_value=corr.fdr(out, 0.05).q_value)


def run_hf_cycle(hf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """H2: panel lead-lag (country + month effects), per-country prewhitened cross-correlation, and Italy's lead against the others."""
    M = H.hf_panel(hf, sorted({i for p in HF_CYCLE_PAIRS for i in p}), countries=[c for c in hf.iso3.unique() if c != "EU27"])
    panel = H.lead_lag_table(M, HF_CYCLE_PAIRS, lags=(1, 3, 6, 12))
    countries = pd.concat([H.country_lead(hf, x, y) for x, y in HF_CYCLE_PAIRS], ignore_index=True)
    italy = []
    for key, g in countries.groupby(["x", "y"]):
        x, y = cast(tuple[str, str], key)
        rk = H.italy_rank(g)
        it = g[g.iso3 == "ITA"]
        if rk and len(it):
            italy.append({"x": x, "y": y, "italy_lag": int(it.lag.iloc[0]), "italy_r": rk["italy_value"], "rank_by_abs_r": rk["rank"], "n_countries": rk["n_countries"],
                          "median_lag_all": float(g.lag.median()), "share_significant": float(g.significant.mean())})
    q = H.hf_panel(hf, ["q_gdp_growth", "q_employment_ths", "q_hours_ths"], countries=[c for c in hf.iso3.unique() if c != "EU27"])
    panel = pd.concat([panel, H.lead_lag_table(q, [("q_gdp_growth", "q_employment_ths"), ("q_gdp_growth", "q_hours_ths")], lags=(1, 2, 4), own_lags=2)], ignore_index=True)
    return {"hf2_panel": panel, "hf2_countries": countries, "hf2_italy": pd.DataFrame(italy)}


def run_hf_passthrough(hf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    windows = {"2005Q1-2021Q2": ("2005-01-01", "2021-06-30"), "2021Q3-2023Q4 (energy shock)": ("2021-07-01", "2023-12-31")}
    panel = pd.DataFrame([{"window": k, **H.passthrough_panel(hf, w)} for k, w in windows.items()])
    by = pd.concat([H.passthrough_by_country(hf, w).assign(window=k) for k, w in windows.items() if k.startswith("2005")], ignore_index=True)
    ranks = []
    for k, g in by.groupby("window"):
        rk = H.italy_rank(g, "pass_through")
        if rk:
            ranks.append({"window": k, **rk, "median_all": float(g.pass_through.median())})
    return {"hf3_passthrough_panel": panel, "hf3_passthrough_countries": by, "hf3_passthrough_italy": pd.DataFrame(ranks)}


def run_hf_events(hf: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    events = italian_events()
    dates = list(zip(events.event, events.date))
    kind = events.set_index("event").type.to_dict()
    series = {"spread_ITA_DEU": H.spread_series(hf, "ITA", "DEU"), "m_long_term_yield": H.wide_hf(hf, "m_long_term_yield")["ITA"],
              "m_esi": H.wide_hf(hf, "m_esi")["ITA"], "m_unemployment_rate": H.wide_hf(hf, "m_unemployment_rate")["ITA"]}
    rows = []
    for name, s in series.items():
        r = H.event_study(s, dates, window)
        rows.append(r.assign(series=name, label=HF_EVENT_SERIES[name], type=r.event.map(kind)))
    out = pd.concat(rows, ignore_index=True)
    return out.assign(q_value=corr.fdr(out, 0.05).q_value).sort_values("p").reset_index(drop=True)


def run_hf_all(hf: pd.DataFrame | None = None, out: Path = OUT) -> dict[str, pd.DataFrame]:
    hf = H.load_highfreq() if hf is None else hf
    out.mkdir(parents=True, exist_ok=True)
    results = {"hf0_stationarity": run_hf_stationarity(hf), "hf1_spread": run_hf_spread(hf), **run_hf_cycle(hf), **run_hf_passthrough(hf), "hf4_events": run_hf_events(hf)}
    for name, df in results.items():
        df.to_parquet(out / f"{name}.parquet")
    return results
