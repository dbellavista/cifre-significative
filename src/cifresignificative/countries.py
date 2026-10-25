"""Country sets used across the project (Eurostat geo codes, ISO3 mapping)."""

EU27 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "HR": "HRV", "CY": "CYP", "CZ": "CZE",
    "DK": "DNK", "EE": "EST", "FI": "FIN", "FR": "FRA", "DE": "DEU", "EL": "GRC",
    "HU": "HUN", "IE": "IRL", "IT": "ITA", "LV": "LVA", "LT": "LTU", "LU": "LUX",
    "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU", "SK": "SVK",
    "SI": "SVN", "ES": "ESP", "SE": "SWE",
}
BENCHMARKS = {"UK": "GBR", "US": "USA", "JP": "JPN"}
AGGREGATES = {"EU27_2020": "EU27"}

# Eurostat geo codes requested from the API
GEO_ALL = [*EU27, *BENCHMARKS, *AGGREGATES]
ISO3 = {**EU27, **BENCHMARKS, **AGGREGATES}
START_YEAR = 2005

NAMES = {
    "AUT": "Austria", "BEL": "Belgium", "BGR": "Bulgaria", "HRV": "Croatia", "CYP": "Cyprus", "CZE": "Czechia", "DNK": "Denmark",
    "EST": "Estonia", "FIN": "Finland", "FRA": "France", "DEU": "Germany", "GRC": "Greece", "HUN": "Hungary", "IRL": "Ireland",
    "ITA": "Italy", "LVA": "Latvia", "LTU": "Lithuania", "LUX": "Luxembourg", "MLT": "Malta", "NLD": "Netherlands", "POL": "Poland",
    "PRT": "Portugal", "ROU": "Romania", "SVK": "Slovakia", "SVN": "Slovenia", "ESP": "Spain", "SWE": "Sweden",
    "GBR": "United Kingdom", "USA": "United States", "JPN": "Japan",
}
