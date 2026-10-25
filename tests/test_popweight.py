"""Population-weighted benchmark on a synthetic panel: heavy countries sit high, so Italy's gap flips sign."""
import pandas as pd

from cifresignificative.analysis import popweight as P

COUNTRIES = ["AUT", "BEL", "BGR", "HRV", "CYP", "ITA"]


def _panel():
    rows = []
    for y in (2021, 2022, 2023):
        for c, v, pop in zip(COUNTRIES, [1, 2, 3, 4, 5, 3.5], [1, 1, 1, 1, 100, 60], strict=True):
            rows += [{"iso3": c, "year": y, "indicator": "x", "value": v}, {"iso3": c, "year": y, "indicator": "population", "value": pop}]
    return pd.DataFrame(rows)


def test_weighted_gap_flips_when_the_heavy_country_is_high():
    r = P.benchmark_gaps(_panel(), ["x"], window=(2021, 2023)).set_index("indicator").loc["x"]
    assert r["median"] == 3.0 and r.wmedian == 5.0
    assert r.gap > 0 > r.wgap and r.flips
    assert P.summary(P.benchmark_gaps(_panel(), ["x"], window=(2021, 2023)))["flips"] == 1
