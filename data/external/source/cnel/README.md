# CNEL renewal dates of national collective agreements

Source: CNEL, *Archivio nazionale dei contratti e degli accordi collettivi di lavoro*, open-data API listed at https://www.cnel.it/Archivio-Contratti-Collettivi/Contratti-Open-Data (IODL 2.0, updated weekly). Queried on 2026-10-17 with `type=ARCHIVIO_CORRENTE` (2,288 agreements) and `type=ARCHIVIO_STORICO` (5,404).
Fields used: `codiceCcnl`, `tipologiaAccordo`, `dataStipula`, `dataDecorrenza`, `dataScadenzaContrattuale`, `partiSindacaliFirmatarie`, `idAccordo`, `titoloCcnl`. Employee counts (`dipendenti`) are empty in every row.

Files (all produced by `uv run python -m cifresignificative.sources.cnel_contracts fetch|build|check`; the raw JSON goes to gitignored `data/raw/cnel/`):
- `data/external/cnel_istat_map.csv`: hand-curated map from each of the 78 Istat agreement leaf groups to one CNEL contract code (`istat_group, istat_name, cnel_code, evidence, note`). Four groups have no code (police, armed forces, fire brigade: pay set by decree after negotiation; Presidenza del Consiglio: own contract not found under one code).
- `data/external/cnel_contracts.csv`: one row per renewal cycle of a mapped code (`code, signed, start, expiry, kind, confederal, n_rows, ids`), built from the raw JSON by rules documented in the module docstring.

## Evidence labels of the map
- `verified`: the title (and, where visible, the employers' association) names the same sector as the Istat group, and renewals signed by CGIL, CISL and UIL federations exist under the code.
- `partly verified`: the contract plausibly covers the group but is wider or narrower than it (the metalworkers' contract for the finer metal groups, the air-transport contract for airport services), several close contracts exist (environmental services, vigilanza), or the archive holds few cycles (schools, journalists, private hospitals).
- `unverified`: no code.
The labels are my reading of titles and signatories on 2026-10-17, not an independent second source (the stricter rule of `rating_actions.csv`); the primary test therefore uses `verified` rows and reports the `partly verified` rows as a sensitivity.

## Known limits
- A code collects side-signings by minor unions on later dates, each with its own expiry (commerce H011: UGL rows of 2017 and 2019). `cnel_contracts.primary_cycles` keeps the cycles signed by CGIL, CISL and UIL when a code has any; C010 (Stellantis, Iveco, CNH, Ferrari: FIOM did not sign) and G031 have none and keep all.
- Public contracts (ARAN codes `S...`) use the date of the *Testo definitivo*, after which the pay rise takes effect; the earlier *Ipotesi* date is not used. Private contracts use the renewal agreement; where only a definitive text exists (C010, AGIDAE schools) it stands for the renewal (`kind`).
- Ten cycles have no start or expiry in the archive (mostly 2021-24 renewals): they carry only the signing date.
- H011 has a 2019-07 definitive text with a four-month period that looks like a minor agreement; I810 (air transport) has many overlapping agreements by section. Neither is corrected by hand.
- The archive's coverage of renewals before 2010 is thin, so cycles are only reliable from about 2010.

## Employee counts (added 2026-10-17)
`workers/` holds three Excel files downloaded by hand from the CNEL archive ("CCNL settore privato" CNEL-INPS lists; employers and employees per agreement code, December 2025, INPS UNIEMENS): `ccnl-di-settore-vigenti-e-ultrattivi.xlsx`, `ccnl-altri.xlsx`, `contratti-aziendali-di-livello-nazionale.xlsx`. `uv run python -m cifresignificative.sources.cnel_workers build` writes `data/external/cnel_workers.csv`. Private sector only (no ARAN agreements); a 2025 snapshot; the other files of the download (managers, collaborators, AEC, new deposits) have no counts and are not kept.
