"""Descriptive income-source comparisons; overlapping frequencies are not populations."""
import pandas as pd


def component_changes(components: pd.DataFrame, deflator: pd.Series, start: int = 2008, end: int = 2024) -> pd.DataFrame:
    """Real amount per recipient and recipient-count growth, on one explicit price basis.

    Component definitions can change across years; this table does not reconcile their
    sum to positive-bracket income or follow the same individuals over time.
    """
    frame = components[components.tax_year.isin([start, end]) & components.component.ne('reddito_complessivo')]
    if frame.duplicated(['comune_id', 'tax_year', 'component']).any():
        raise ValueError('Duplicate component observations')
    w = frame.pivot(index=['comune_id', 'component'], columns='tax_year', values=['amount_nominal_eur', 'recipient_count'])
    amount, count = w['amount_nominal_eur'], w['recipient_count']
    per = amount / count.where(count.gt(0))
    growth = 100 * (per[end] / per[start] / (deflator.loc[end] / deflator.loc[start]) - 1)
    out = growth.rename('real_per_recipient_growth_pct').to_frame()
    out['recipient_growth_pct'] = 100 * (count[end] / count[start].where(count[start].gt(0)) - 1)
    out = out.reset_index()
    peers = out[out.comune_id.ne('040007')].groupby('component').real_per_recipient_growth_pct.agg(['median', 'count'])
    return out.merge(peers.rename(columns={'median': 'peer_median_growth_pct', 'count': 'peer_n'}), on='component', validate='many_to_one')
