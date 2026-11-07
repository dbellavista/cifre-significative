# Download protocol

Order of attempts for any data source. Stop at the first that works; record the URL and date in `data/raw/manifest.json`
(`cifresignificative.core.sources._common.record`).

1. **Official API or bulk file.** Eurostat Statistics API, World Bank `wbgapi`, IMF DataMapper, OECD SDMX (`sdmx.oecd.org`), AMECO zip, EC Excel tables. Preferred: exact, reproducible, polite.
2. **Plain HTTP to a stable URL.** Some sites block page URLs but serve files (oecd.org returns 403 to scripts, yet `https://stat.link/files/<doi-id>/<id>.xlsx` StatLinks download fine; the StatLink IDs can be found in an archived copy of the page on web.archive.org). Commission "docsroom" documents serve PDFs directly (text PDFs: use `pypdf`, no OCR needed; scanned PDFs with no text layer: PaddleOCR via `pensieri-ocr`, `sources/ocr.py`, skill `cifresignificative-workflow-ocr-scanned-pdf`, never Tesseract).
3. **Headless browser that reads the page's own data requests** (`src/cifresignificative/core/sources/browser.py`, Playwright + Chromium). Open the page, capture the JSON/XHR responses, decode them. Real browsers pass checks that scripts do not (oecd.org pages load headless). Example: `sources/procurement.py` opens the Single Market Scoreboard, finds the embedded public Power BI report, walks its 11 pages and decodes the `querydata` responses (`sources/powerbi.py`).
4. **Visible browser with a human step** (`cifresignificative.core.fetch --interactive`, `browser.handoff`). Used only if a page needs a person (a check the user must pass). The profile is persistent (`data/.browser_profile/`, gitignored) so a solved check is remembered. Never automate solving a check.

A person's own browser session can stand in for step 4: ask them to open the page, use the browser's developer tools to "Save all as HAR" (or export the one response), and read only the data response from it. Do not read cookies, tokens or other requests from the file.

## Rules
- Public data only; respect each site's terms, rate limits and attribution. Slow, single-threaded requests.
- Never commit cookies, tokens or the browser profile. Never read a browser's own cookie store without an explicit, narrowly scoped request, and stop if the environment blocks it.
- If a site cannot be reached by 1-4, fall back to a manual download by a person (`docs/manual_data.md`); keep the downloaded file under `data/external/source/` with its provenance.
- Verify every extracted series against a known anchor (a figure in the source, an official country note) before it enters the panel, and add a test.

## Status per source
| source | route used |
|---|---|
| Eurostat, World Bank, IMF, AMECO, OECD SDMX | 1 |
| EC VAT gap Excel | 2 (URL on the Commission site) |
| PISA 2022 | 2 (StatLink found in a web.archive.org copy of the annex page) |
| Procurement indicators (Single Market Scoreboard) | 3 (Power BI responses) |
| PIAAC 2023 | route 4 variant: the endpoint is behind a Cloudflare challenge that a script cannot pass, so a person saved the page's network requests (HAR) from their own browser; only the data response body was read (`scripts/import_piaac.py`) |
| MEF IRPEF return statistics | route 2: public GET export links, found by crawling the portal's paginated listing (`sources/mef_portal.py`, `scripts/fetch_mef_irpef.py`), one request per second, CC BY 3.0 IT |
| Daily 10-year yields (Italy, Germany, Spain, France, Portugal, Greece) | manual export from Investing.com (no free API serves country-level daily yields: the ECB Data Portal has only euro-area curves, FRED is monthly, Stooq blocks scripts); files in `data/external/source/daily_yields/`, loader `sources/daily_yields.py` |
| Italian rating changes | Moody's and Fitch from the agencies' own rating-history pages (saved by hand as rendered HTML on 2026-10-15), DBRS from its site API (`/api/issuer/documents` and `/api/document/rating-view`, with a browser session cookie), all parsed to `data/external/source/rating_history/`; S&P from its Italy entity page (table "Related Credit News and Research", two pages saved by hand); `data/external/rating_actions.csv` with an evidence label per row; four dates (2017-2018) from the Treasury public-debt reports read with `pdftotext -layout` |
| ECB Securities Holdings Statistics | route 1 with a twist: dataflow `SHS` is blocked by the ECB firewall (HTTP 400 on data queries), the sector-level `SHSS` answers to a full 18-position key (`sources/ecb_shss.py`) |
| EBA transparency exercise (sovereign exposures) | route 2: `tr_sov.csv` per year is a direct link on each year's page (`sources/eba_sovereign.py`; 2020 from tools.eba.europa.eu; files of 75-111 MB, gitignored) |
| Scanned PDFs (Comune consolidated accounts 2019) | route 2 for the file, then OCR: `pensieri-ocr` shell command on the PaddleOCR server of the sister project, run natively (`sources/ocr.py`), token cache in `data/raw/ocr/`, checked against the balance-sheet identities, and a hand-review file in `data/external/ocr_review/` when they fail (`analysis/consolidato.py`, `tests/core/test_ocr.py`) |
