"""Reproduce income comparisons under frozen and direct-index price adjustments.

Run from the project root. Outputs are provisional diagnostics, not publication claims.
"""
import json
from pathlib import Path

import pandas as pd

from cifresignificative.studies.cesena.analysis.income_followup import component_changes

BASE = Path('data/raw/cesena_income_baseline')
OUT = Path('studies/cesena/reports/income_evidence')


def main() -> None:
    panel = pd.read_parquet(BASE / 'irpef_summary.parquet')
    prices = pd.read_csv(OUT / 'deflator_audit.csv').set_index('year')
    levels = panel[panel.year.isin([2008, 2018, 2024])][['comune_id', 'comune', 'year', 'mean_income_eur', 'mean_income_real_eur', 'median_income_eur', 'taxpayers_pos']].copy()
    levels['mean_real_direct_hicp_eur'] = levels.mean_income_eur / levels.year.map(prices.eurostat_hicp_2020_1)
    levels.to_csv(OUT / 'review_levels.csv', index=False)
    rows = []
    for lo, hi in [(2008, 2024), (2008, 2018), (2018, 2024)]:
        a, b = levels[levels.year.eq(lo)].set_index('comune_id'), levels[levels.year.eq(hi)].set_index('comune_id')
        for measure in ['mean_income_eur', 'mean_income_real_eur', 'mean_real_direct_hicp_eur', 'median_income_eur']:
            g = 100 * (b[measure] / a[measure] - 1)
            for cid, value in g.items():
                rows.append({'comune_id': cid, 'comune': a.loc[cid, 'comune'], 'start': lo, 'end': hi, 'measure': measure, 'growth_pct': value, 'rank': g.rank(ascending=False).loc[cid], 'peer_median_growth_pct': g.drop('040007').median(), 'cities': len(g.dropna())})
    pd.DataFrame(rows).to_csv(OUT / 'review_growth.csv', index=False)
    c = pd.read_parquet('data/processed/cesena/income_followup/income_components.parquet')
    component_changes(c, prices.panel_chained_2020_1).to_csv(OUT / 'component_changes.csv', index=False)
    c['income_per_recipient_nominal'] = c.amount_nominal_eur / c.recipient_count.where(c.recipient_count.gt(0))
    c = c[c.tax_year.isin([2008, 2018, 2024])]
    c.to_csv(OUT / 'review_component_endpoints.csv', index=False)
    raw = json.loads((BASE / 'eurostat_hicp_annual_20261106.json').read_text())
    indices = pd.Series({int(y): raw['value'][str(i)] for y, i in raw['dimension']['time']['category']['index'].items()})
    rates = json.loads((BASE / 'eurostat_hicp_rates_review.json').read_text())
    published = pd.Series({int(y): rates['value'][str(i)] for y, i in rates['dimension']['time']['category']['index'].items()})
    audit = pd.DataFrame({'index': indices, 'rate_implied_by_index_pct': indices.pct_change() * 100, 'published_rate_pct': published})
    audit['rate_min_from_rounded_indices'] = 100 * ((indices - .05) / (indices.shift() + .05) - 1)
    audit['rate_max_from_rounded_indices'] = 100 * ((indices + .05) / (indices.shift() - .05) - 1)
    audit['rounding_intervals_overlap'] = ((audit.rate_max_from_rounded_indices >= published - .05) & (audit.rate_min_from_rounded_indices <= published + .05)).where(indices.shift().notna())
    audit.rename_axis('year').to_csv(OUT / 'review_hicp_consistency.csv')
    print(pd.DataFrame(rows).query("comune_id == '040007'").to_string(index=False))


if __name__ == '__main__':
    main()
