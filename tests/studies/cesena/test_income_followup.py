import pandas as pd
import pytest

from cifresignificative.studies.cesena.analysis.income_followup import component_changes


def test_component_growth_separates_prices_and_recipients():
    rows = []
    for cid in ['040007', '040012']:
        for year, amount, count in [(2008, 1000, 10), (2024, 2400, 20)]:
            rows.append({'comune_id': cid, 'tax_year': year, 'component': 'employment', 'amount_nominal_eur': amount, 'recipient_count': count})
    result = component_changes(pd.DataFrame(rows), pd.Series({2008: 1., 2024: 1.2}))
    assert result.real_per_recipient_growth_pct.tolist() == [0., 0.]
    assert result.recipient_growth_pct.tolist() == [100., 100.]
    assert result.peer_n.tolist() == [1, 1]


def test_duplicate_components_rejected():
    row = {'comune_id': '040007', 'tax_year': 2008, 'component': 'employment', 'amount_nominal_eur': 1000, 'recipient_count': 10}
    with pytest.raises(ValueError, match='Duplicate'):
        component_changes(pd.DataFrame([row, row]), pd.Series({2008: 1., 2024: 1.2}))
