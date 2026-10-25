# Eurosystem public-sector purchases by country (ECB)

Two CSV exports saved by hand from the ECB asset-purchase-programme pages (Monetary policy, asset purchase programmes; "PSPP breakdown history" and "PEPP public sector securities breakdown history"), downloaded 2026-10-15, data to 2026-09. Kept as received.
- `PSPP_breakdown_history.csv`: PSPP, 2015-03 to 2026-09. Blocks: monthly net purchases per country (EUR million, net of redemptions; last column = ECB cumulative), WAM of the portfolio, WAM of the eligible universe (published monthly from 2021-08).
- `PEPP_public_sector_securities_breakdown_history.csv`: same layout for PEPP, 2020-03 to 2026-09 (Greece included).
- Not used (other ECB exports in the same download): `PEPP_purchase_history`, `PEPP_redemptions_history` (euro-area aggregates), `PEPP_private_sector_*` (corporate, covered, ABS, commercial paper).
- Flows, not stocks: `stock` in the loader (`cifresignificative.sources.eurosystem_holdings`) is the cumulative sum of net purchases at amortised cost; the monthly figures sum to the ECB cumulative column within rounding (max difference 6, Italy 1). At end-2025 it gives Italy PSPP EUR 312.2bn and PEPP 246.8bn, the figures of the Treasury's 2025 debt report, so the two sources share a basis (the ECB cumulative column in the files, 277,960 and 219,781, is the September 2026 value). It is not a market value.
- Cells use thousands separators ("-2,957") and two date formats; the loader handles both. Refresh: replace the files from the ECB pages.
