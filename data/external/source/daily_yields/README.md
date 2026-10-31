# Daily 10-year government bond yields (Italy, Germany, Spain, France, Portugal, Greece)

Manual export from Investing.com ("<Country> 10-Year Bond Yield Historical Data", daily, downloaded 2026-10-14). The site caps an export at 5,000 rows, so each country comes in two files that overlap (Italy and Germany in early 2023, Spain, France, Portugal and Greece in 2022-2023; the overlaps are identical, the loader raises otherwise). `Price` is the daily close, in percent. Dates are `MM/DD/YYYY`. Files are named `<country>_10y_<span>.csv`. Loader: `src/cifresignificative/core/sources/daily_yields.py`.
Provenance caveat: a commercial aggregator, not an official series; the monthly Eurostat convergence yield (`m_long_term_yield`) is the cross-check (monthly means: correlation 0.9994-1.0000 for Spain, France and Portugal, 0.979 for Greece).
The exports contain indicative weekend quotes (Italy 370, Germany 182, Spain 358, France 280, Portugal 431, Greece 324 rows, mostly 2013-2023); the loader drops them.
Greece: October 2011 to January 2012 monthly means are 6-12 pp above the Eurostat yield (the restructuring period; the aggregator quotes a different benchmark bond): do not use the Greek daily series for that window without checking.
Source holes: Greece has no quotes from late February to 23 June 2005 (121 days); Portugal has one 7-day gap in June 2007.
