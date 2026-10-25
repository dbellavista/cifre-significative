# Filling the manual series by hand

Two education series have no API: **PISA 2022** (the World Bank copy stops at 2018) and **PIAAC** adult skills.
Templates live in `data/external/` (tracked in git). Rows with an empty `value` are ignored, so you can fill
them in gradually; a filled row **must** also have a `source`. Then rebuild:

```
uv run python -m cifresignificative.clean.build
uv run pytest
```

## Outstanding manual downloads (status 2026-10-14)

| Item | State | Who and how |
|---|---|---|
| Daily 10-year yields: Spain, France, Portugal, Greece | **DONE** (2026-10-14) | files in `data/external/source/daily_yields/` (Investing.com, two chunks per country because of the 5,000-row cap; overlaps agree) |
| ECB SHSS holdings; EBA transparency 2020-2025 | **DONE**, automated | `uv run python -m cifresignificative.fetch --only ecb_shss eba_sovereign` (see the sections below) |
| **Italian rating dates** | **OPEN, highest value** | see "Rating actions" below |
| Eurosystem holdings of Italian debt | **DONE** (2026-10-15), manual download | ECB exports `PSPP_breakdown_history.csv` and `PEPP_public_sector_securities_breakdown_history.csv` (monthly net purchases and WAM per country, to 2026-09) in `data/external/source/eurosystem_holdings/` (README); loader `sources/eurosystem_holdings.py`; flows net of redemptions, the stock is their cumulative sum at amortised cost (equals the Treasury's end-2025 figures); the other three ECB exports (aggregates, private sector) are not needed |
| MEF IRPEF: `AE*` tables by prevalent income, VAT turnover of flat-rate filers | **AE tables by turnover class DONE** (2026-10-15, 84 files, `data/external/source/mef_irpef_fatturato/`, loaded in `mef_irpef_dims.parquet`, used in notebook 32: professionals' cost share by turnover band); `comune`: the portal lists no tables; `AE*` by activity and age are in the catalogues, not downloaded | `uv run python scripts/fetch_mef_irpef.py get <name> --tables '^AE'` |
| CompNet (firm-level productivity dispersion, 10th vintage) | **NOT AVAILABLE** (access is for researchers with an institutional affiliation). Replaced by the Eurostat size-class statistics (notebook 42), which have no dispersion inside classes | free access on request: form at https://www.iwh-halle.de/en/about-the-iwh/research-centres/research-data-centre/compnet-database/request-form (variable list: `CompNet_variables_10th_vintage.xlsx` on the vintage page); save the files to `data/external/source/compnet/` and describe the use |
| CNEL renewal dates of national collective agreements | **DONE** (2026-10-17), automated, no manual download | CNEL open-data API (`https://www.cnel.it/Archivio-Contratti-Collettivi/Contratti-Open-Data`): `uv run python -m cifresignificative.sources.cnel_contracts fetch`, then `build` after editing `data/external/cnel_istat_map.csv`; README with the evidence rubric in `data/external/source/cnel/` |
| EBA transparency 2013-2018 | OPTIONAL | other file layout; only for a longer bank-exposure series |

### Rating actions (`data/external/rating_actions.csv`)
Goal: dated Italian sovereign rating and outlook changes (not confirmations, not bank ratings), each with two independent sources that give the **same day**. Currently 40 rows, **all `verified`** against the agencies' own pages (S&P, Moody's, Fitch, DBRS; 2026-10-15; see `data/external/source/rating_history/README.md`: how the data were captured and what was left out). Only Scope Ratings is not covered.
1. Sources: the agencies' own press releases (free registration at spglobal.com/ratings, ratings.moodys.com, fitchratings.com, dbrs.morningstar.com, scoperatings.com; search "Italy" and open the sovereign actions) and the Treasury (MEF) press releases at mef.gov.it (search "rating"). The Treasury's public-debt reports (`archive_presentazioni/elem_0008.html`) were checked: only the 2017 and 2018 reports carry dates, and they are already used. Aggregators (countryeconomy.com, fxempire.com, theglobaleconomy.com) count as one source each and sometimes disagree by days.
**Update 2026-10-15:** every row is now `verified` against an agency's own record, so the aggregator-only concern below is closed (it is kept for the history of the question). **Source quality (checked 2026-10-14; reviewer suggestions tested).** Aggregators probably share an upstream source, so two aggregators agreeing is weak evidence: tradingeconomics.com (public page, about 30 dated rows from 2005 to 2026, **S&P and DBRS only**, no Moody's or Fitch) lists the same dates as theglobaleconomy.com, including the Saturday 12 April 2025 that fxempire also lists. Seven `partly verified` rows rest only on two aggregators (fxempire and countryeconomy) and have no agency, regulator, Treasury or news-article source: Fitch 2020-04-28 and 2021-12-03, S&P 2021-10-22, 2022-07-26 and 2025-04-11, Moody's 2023-11-17 and 2025-05-23. Their labels are kept (changing them would change the notebook 27 results), but a document from an agency or the Treasury is needed to make them solid; the pooled test has been re-run by source type (notebook 27, D2c, `experiments.run_daily_rating_support`): the weak one-day tilt of the 17 `partly verified` events comes entirely from these seven (one-day mean signed z 0.91, p 0.011, all 7 in the expected direction), the other 10 show none (-0.19, p 0.73); the seven are all from 2020 on, so source and period are confounded, and documents for them would settle it. Not usable: worldgovernmentbonds.com (the Italy page has no dated history, only current ratings). Unconfirmed suggestions, to try before relying on them: the agencies' "Rating History" pages (login needed; whether the full history can be exported on a free account is unknown), ESMA's rating disclosures (no register called `esma_registers_radar` was found on registers.esma.europa.eu), and SEC Rule 17g-7 rating-history files on each agency's regulatory-disclosure page (probably limited to issuer-paid ratings plus a sample, so Italy may be missing). Context: since Regulation (EU) 462/2013 (CRA III) EU-registered agencies publish sovereign rating calendars in advance, so actions from about 2014 fall on announced dates that markets may anticipate; before that dates were ad hoc. That regulation does not explain why the Treasury reports lack a history (they simply carry no table).
2. Gaps to fill: none among the four agencies for 2011-09-19 onward (S&P's 2013 downgrade is dated 2013-07-10, the publication of its correction, confirmed by the owner). Scope Ratings has no Italian rating in the sources checked. Pre-2011 actions (S&P 2011-05-20, 2006-10-19; Fitch 2005-06-29, 2006-10-19; DBRS 2011-02-03) are deliberately outside the file.
3. Row format (one row per agency action): `date` (YYYY-MM-DD), `agency` (S&P, Moody's, Fitch, DBRS, Scope), `action` (`downgrade`, `upgrade` or `outlook`), `rating_from`, `rating_to`, `outlook_from`, `outlook_to` (`negative`/`stable`/`positive`; for an `outlook` action both are required), `evidence` (`verified` (the day is in the agency's own record of the action), `partly verified` (two other independent sources give the same day) or `unverified`), `sources` (URLs separated by ` | `), `notes`. Non-trading days snap to the next trading day; weekends are not announcement dates, so a Saturday date is probably wrong.
4. Then: `uv run inv test.run -f tests/test_daily.py`, re-run `experiments.run_daily_all`, re-execute notebook 27 and re-check every number in its takeaways, `reports/findings.md` (item 27) and `docs/analysis_framework.md`.
Another person can paste the dated list (or the pages' text) into the chat and the rows can be written and checked from it.

## MEF IRPEF return statistics (`data/external/source/mef_irpef/`): DONE, manual export
Exported by a person from the Department of Finance open-data portal (`https://www1.finanze.gov.it/finanze/analisi_stat/public/index.php?opendata=yes`, "Classi di reddito" / "Persone fisiche" / "Persone fisiche titolari"); `download.sh` (the curl loop used), `mapping.csv` (table code -> description) and one CSV per filing year and table are kept as received, except that the 77 empty files (0 bytes: filing years 2005-2008, the `AB*0203/0303/0202` tables of 2009-2011, `AAPFDIP0202` and `AAPFPEN0202` of 2016) were not copied. A re-request of the 2016 files and of 2005/2008/2009 samples returned empty again, so they are not transient; 2016 `AAPFDIP0201` exists but is by region, not by income class. The file prefix is the **filing** year. Loader: `cifresignificative.sources.mef_irpef` (rebuild with `uv run python -m cifresignificative.clean.build`). To refresh: rerun `download.sh` in a new folder, copy the non-empty files under `data/external/source/mef_irpef/<year>/`, rebuild.

**Automated since 2026-10-14:** `uv run python scripts/fetch_mef_irpef.py list <name>` shows the portal's catalogue of tables for a classification (`classi_reddito`, `ventili`, `regione`, `attivita`, `eta`, `fatturato`, `comune`) and `get <name> [--tables REGEX] [--years A B]` downloads individuals' tables (default `^(AAPF|ABPF|AE)`, narrowed to `^(AAPF|ABPF)` for the batches in `data/external/source/mef_irpef_*`) at one request per second into `data/external/source/mef_irpef_<name>/`; then rebuild. The portal data are CC BY 3.0 IT: cite the Ministry of Economy and Finance, Department of Finance. Not downloaded: `comune` (municipal), the `AE*` tables by prevalent income, companies, VAT and IRAP.

## Rules for every CSV
- Columns: `iso3,year,indicator,value,theme,source`. Don't rename columns or indicators; don't add rows for countries
  outside the template (the loader rejects unknown `iso3`).
- `value`: plain number, dot as decimal separator, no units or `%`.
- `source`: table name plus URL or page, e.g. `PISA 2022 Vol I Table I.B1.2.1`. It is stored as `manual:<source>`.
- If a country did not take part, leave `value` empty (do not enter 0 or NA).
- Don't hand-edit anything under `data/processed/`; always rebuild.

## 1. `pisa_2022.csv` (mean scores, 2022 wave): DONE, automated
Filled by `uv run python scripts/import_pisa_2022.py` from the OECD workbook `data/external/source/pisa2022_ch2_tables.xlsx`
(StatLink https://stat.link/files/53f23881-en/xmrlsh.xlsx, chapter 2 tables I.B1.2.1-3). oecd.org itself blocks scripted downloads (HTTP 403)
but `stat.link` serves the Excel files directly; an archived copy of the annex page (web.archive.org) was used to find the StatLink IDs.
The manual steps below are kept as a fallback if the workbook changes.

Original manual instructions:
Indicators `wb_pisa_math`, `wb_pisa_reading`, `wb_pisa_science` keep the same names as the World Bank series (2006-2018)
so the history continues in one series.
1. Open the OECD *PISA 2022 Results (Volume I)*, Annex B1, "Results for countries and economies":
   https://www.oecd.org/en/publications/pisa-2022-results-volume-i_53f23881-en/full-report/component-25.html
2. Take the **mean score** column from Table I.B1.2.1 (mathematics), I.B1.2.2 (reading), I.B1.2.3 (science). The tables
   can also be downloaded as Excel from the same page.
3. Copy one number per country into the matching rows (`iso3` uses ISO3: Greece = GRC, UK = GBR). Use the "Mean" column,
   not the standard error or percentile columns. Country notes are a handy cross-check
   (https://www.oecd.org/en/publications/pisa-2022-results-volume-i-and-ii-country-notes_ed6fbcc5-en.html).
4. Caution: PISA 2022 scores dropped almost everywhere, partly due to the pandemic; don't read 2018 -> 2022 as a long-run trend.
   Where a table carries a footnote on a country (e.g. a data-quality caveat), copy the number anyway and add the footnote text to `source`.

## 2. `piaac.csv` (Survey of Adult Skills): DONE, from a browser export
The OECD Education GPS data endpoint is behind a Cloudflare challenge for scripts, so a person opened
https://gpseducation.oecd.org/IndicatorExplorer?plotter=h5&query=51 in their own browser and saved the page's requests as a HAR file;
the table body (a tab-separated file with ~66 indicators per country: means by age, sex, education, parental education, levels, change 2012-2023)
is saved in `data/external/source/piaac/gpseducation_piaac2023.tsv` and `scripts/import_piaac.py` turns it into `data/external/piaac.csv`.
Only the response body was read from the HAR (no headers or cookies). 2012 levels are the 2023 level minus the OECD's reported change.
To refresh: repeat the browser export, replace the TSV, rerun the script and `cifresignificative.clean.build`.

Original manual instructions (kept as a fallback):
Indicators:
| indicator | meaning | year rows |
|---|---|---|
| `piaac_literacy` | mean literacy proficiency score (adults 16-65) | 2012 (Cycle 1) and 2023 (Cycle 2) |
| `piaac_numeracy` | mean numeracy score | same |
| `piaac_low_literacy_pct` | % of adults at Level 1 or below in literacy | same |

Cycle 1 was fielded in several rounds (2011-12 for most countries, later rounds for some, around 2014-15): check the fieldwork year
in the OECD documentation and enter it in `year`; the template pre-fills 2012, edit the year where a country differs (the loader accepts
any year). Cycle 2 was fielded in 2022-23 (enter 2023).
1. Open the OECD *Survey of Adult Skills 2023* report and its Data Explorer, via
   https://www.oecd.org/en/publications/survey-of-adult-skills-2023_3639d1e2-en.html
2. In the Data Explorer pick *Literacy* -> "Mean proficiency scores" by country, then repeat for *Numeracy*, and for the
   percentage of adults at "Level 1 or below".
3. Participation differs by cycle and not every EU27 country took part: if a country is absent from the OECD table for a cycle,
   leave its `value` empty.
4. Cycle 1 and Cycle 2 literacy scales are linked, so the change is meaningful, but compare only countries where
   the OECD marks the difference as significant.

## 3. `procurement.csv` (public procurement competition)
Indicators `proc_single_bidder_pct` (share of contracts awarded with a single bidder; framework agreements and direct awards excluded)
and `proc_direct_awards_pct` (share of procedures awarded without a call for tenders). Both in % of procedures, per EU27 member state.
The template pre-fills 2015, 2019 and 2023; add rows for other years if you want a denser series.
1. Open the Single Market Scoreboard "Public procurement" page (Italian version works too):
   https://single-market-scoreboard.ec.europa.eu/node/264_it
2. Each indicator has a chart/table by member state and year (the page is interactive: the numbers are not in the page source,
   so read them off the chart tooltips or use the "download" option if the page offers one). Source note: the data come from TED.
3. Enter the percentage (e.g. `18.4`) for each country/year; leave empty if the scoreboard shows no value.
4. Use `source` = `Single Market Scoreboard public procurement <year>`.
Thresholds the scoreboard uses for its colour code: single bidder green <= 10%, red > 20%; direct awards green <= 5%, red >= 10%.

## 4. Istat non-observed economy workbook: DONE, from a browser download
Not a CSV for `data/external/`: the Excel file "Tavole-Economia-non-osservata" (2025 edition, years 2011-2023) comes from the Istat release page (`https://www.istat.it/comunicato-stampa/economia-non-osservata-nei-conti-nazionali-anno-2023/`, section "Tavole") and is stored as `data/external/source/istat_unobserved/Tavole-Economia-non-osservata_2025.xlsx`. `sources/istat_unobserved.py` reads sheets TAVOLA 1 and TAVOLA 3 by position, so for a new edition check that the layout is unchanged.

## Check before committing
- Italy PISA 2018 math in the series is 487 (a sanity anchor for scale); entered 2022 scores should be of similar magnitude (300-600).
- `uv run pytest` still passes and `uv run python -m cifresignificative.clean.build` prints no `manual:` error.
- Commit the filled CSVs (`git add data/external`), message e.g. `data: add PISA 2022 and PIAAC values`.

## ECB Securities Holdings Statistics (`SHSS`): API works, no manual export needed
`SHS` is blocked by the ECB firewall, but the sector-level `SHSS` dataflow is served by `data-api.ecb.europa.eu` with a full 18-position key. History starts in 2021-Q1. Banks (`S122`) against issuer `S13` are not published; use `S12` or `S121` against `S13`, or use the residual `S1 - S12 - S1M` (government plus non-financial corporations, an upper bound; `S11` is not published). For bank-by-sovereign exposures use the EBA transparency exercise: the full-database files are direct links, no manual step. Downloaded 2026-10-14 to `data/raw/eba_transparency/` (gitignored, 75-111 MB each): `<year>_tr_sov.csv` for 2020-2025 (`https://www.eba.europa.eu/assets/TE<year>/Full_database/<id>/tr_sov.csv`; 2020 from `tools.eba.europa.eu`) and `TR_Metadata.xlsx` for 2021, 2024, 2025. Semi-annual (Dec and Jun) from 2019-12 to 2025-06; about 12 Italian banks (`NSA=IT`); `Country` is a numeric code defined in `TR_Metadata.xlsx`; the 2020 file is latin-1 encoded. Loader: `sources/eba_sovereign.py` (also in `cifresignificative.fetch`), tests in `tests/test_eba_sovereign.py`.

### Budget-law dossiers and deficit targets (2026-10-17)
`uv run python -m cifresignificative.sources.budget_laws fetch` downloads the dossiers of the Camera (documenti.camera.it) and the 2018 copy hosted by Astrid; `senato.it` answers scripts with an empty HTTP 202 page, so a Senate document is a manual download into `data/raw/budget_laws/` (2022: `S18_AS2448_nota274.pdf`, the Nota di lettura 274, used for its net summary only). To add a year: add the dossier to `DOSSIERS`, run `... budget_laws check` to see its tables and identities, add the row(s) to `data/external/budget_laws_tables.csv` (table = n-th composition table, `title_contains` guards it), then `... build`. The programmatic targets: `uv run python -m cifresignificative.sources.deficit_targets fetch|build` (add a draft budgetary plan to `DPB` or a dossier to `DOSSIER_TARGETS`). Both parsers need poppler's `pdftotext`; the committed CSVs are the source of truth, so the build and CI never parse.

