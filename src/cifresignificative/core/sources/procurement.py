"""EU Single Market Scoreboard public-procurement indicators (DG GROW, from TED), read from the scoreboard's embedded Power BI report.

The scoreboard page embeds a public Power BI report; its indicator pages request country x year values, which we capture with a
headless browser and decode (`powerbi.py`). Years 2021-2024 are available.
"""
import json
import re
import urllib.parse
from pathlib import Path

import pandas as pd

from cifresignificative.core.countries import ISO3
from cifresignificative.core.sources import browser as br
from cifresignificative.core.sources._common import record
from cifresignificative.core.sources.powerbi import decode_matrix

PAGE = "https://single-market-scoreboard.ec.europa.eu/node/264_it"
RAW = Path("data/raw/procurement/scoreboard_powerbi.json")
# report measure number -> (panel name, multiplier)  [shares are stored as fractions in the report]
INDICATORS = {
    "01": ("proc_single_bidder_pct", 100), "02": ("proc_direct_awards_pct", 100), "03": ("proc_publication_value_pct_gdp", 100),
    "05": ("proc_price_only_award_pct", 100), "06": ("proc_decision_speed_days", 1), "07": ("proc_sme_contracts_pct", 100),
    "08": ("proc_sme_bids_pct", 100), "09": ("proc_lots_pct", 100),
}


def _capture(headed: bool) -> dict:
    pages: dict[int, list[dict]] = {}
    with br.browser(headed=headed) as ctx:
        page = ctx.new_page()
        page.goto(PAGE, wait_until="networkidle", timeout=90000)
        if br.is_blocked(page) and headed:
            br.handoff(page, "The scoreboard shows an access check.", ready=lambda p: not br.is_blocked(p))
        embed = None
        for f in page.frames:
            if "oriurl=" in f.url:
                m = re.search(r"oriurl=([^&]+)", f.url)
                assert m is not None
                embed = urllib.parse.unquote(m.group(1))
        if not embed:
            raise RuntimeError("Power BI embed not found on the scoreboard page")
        report = ctx.new_page()
        current = [1]
        report.on("response", lambda r: pages.setdefault(current[0], []).append(r.json()) if "querydata" in r.url else None)
        report.goto(embed, wait_until="networkidle", timeout=90000)
        report.wait_for_timeout(8000)
        n, total = 1, None  # the first report page shows no "n of m" counter
        while True:  # walk through every report page using the counter
            m = re.search(r"(\d+) of (\d+)", report.inner_text("body"))
            if m:
                n, total = int(m.group(1)), int(m.group(2))
            if total is not None and n >= total:
                break
            current[0] = n + 1
            report.get_by_role("button", name=re.compile("next page", re.IGNORECASE)).first.click(timeout=30000)
            report.wait_for_timeout(5000)
    return {"embed": embed, "pages": pages}


def fetch(force: bool = False) -> Path:
    if RAW.exists() and not force:
        return RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = _capture(headed=False)
        if not data["pages"]:
            raise RuntimeError("no data requests captured")
    except Exception:
        if not br.INTERACTIVE:
            raise
        data = _capture(headed=True)
    RAW.write_text(json.dumps(data))
    record("procurement", source="single-market-scoreboard-powerbi", dataset="public procurement indicators 1-9", url=PAGE)
    return RAW


def load() -> pd.DataFrame:
    raw = json.loads(RAW.read_text())
    iso_of = dict(ISO3)
    frames = []
    for responses in raw["pages"].values():
        for resp in responses:
            d = decode_matrix(resp)
            if d.empty or d.year.nunique() < 2:
                continue
            col = d.columns[2]  # first measure = the yearly value; later ones are "latest/most recent value" copies
            m = re.search(r"Indicators\.(\d+) ", col)
            if not m or m.group(1) not in INDICATORS:
                continue
            name, mult = INDICATORS[m.group(1)]
            sub = d[[d.columns[0], "year", col]].rename(columns={d.columns[0]: "geo", col: "value"})
            sub = sub.assign(value=pd.to_numeric(sub.value, errors="coerce")).dropna(subset=["value"])
            sub = sub[sub.geo.isin(iso_of)]
            frames.append(sub.assign(value=sub.value * mult, indicator=name, iso3=sub.geo.map(iso_of), flag=None, theme="governance",
                                     source="single-market-scoreboard:TED", year=sub.year.astype(int)))
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["indicator", "geo", "year"])
    return out[["iso3", "geo", "year", "indicator", "value", "flag", "theme", "source"]]
