"""Cell definitions of the notebooks of this study: NOTEBOOKS[name] = [("md"|"code", text), ...], one module per notebook.

Read by scripts/make_notebooks.py; the first cell of every notebook is SETUP."""
import importlib

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {}
M, C = "md", "code"
LAST = "LAST = 2024  # last year with broad coverage (2025 is provisional for many series)\n"
# prepended to the "figures quoted in the takeaways" cell: every number of the takeaways must be printed by a cell (tests/studies/italy_productivity/test_takeaways.py)
QUOTED = '''# Figures quoted in the takeaways: Italy, median of the other members, rank (1 = highest) among the members with a value
def quoted(ind, years, countries=()):
    w = wide(panel, ind, EU27_ISO3)
    for y in years:
        r = w.loc[y].dropna()
        print(f"{ind} {y}: Italy {r[ITALY]:.2f} | median of the others {r.drop(ITALY).median():.2f} | rank {int(r.rank(ascending=False)[ITALY])}/{len(r)}" + "".join(f" | {c} {r[c]:.2f}" for c in countries))  # two decimals: a takeaway rounded to one is then checked without ties
'''
SETUP = '''import os, warnings
from pathlib import Path
while not Path("pyproject.toml").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir(Path.cwd().parent)  # run from the project root so data/ paths resolve (the notebook sits in studies/<name>/notebooks)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
from cifresignificative.studies.italy_productivity.analysis.data import *
from cifresignificative.core.viz.style import *
apply_style()
panel = load_panel()
'''

for _name in ['01_coverage_eda', '02_italy_vs_eu', '03_debt_investment', '04_demography_pensions', '05_education_labour', '06_correlations', '07_regional_italy', '08_productivity_sectors', '09_governance_evasion', '10_spending_composition', '11_wages_gender', '12_higher_education', '13_industry_detail', '14_ageing_debt_scenarios', '15_spending_efficiency', '16_long_history_counterfactual', '17_wages_productivity', '18_gender_childcare_tax', '19_pension_reform_event_study', '20_regional_growth_robustness', '21_crime_migration', '22_governments_policies', '23_growth_accounting', '24_correlation_discovery', '25_regional_correlation_discovery', '26_high_frequency_dynamics', '27_daily_spreads_ratings', '28_industry_panel', '29_latent_structure', '30_income_distribution', '31_self_employment', '32_self_employed_income', '33_irpef_regions_sectors_ages', '34_regional_south', '35_regulation_productivity', '36_wage_setting', '37_nuts_spatial', '38_budget_laws', '39_contract_timing', '40_eu_funds_nuts1', '41_budget_placebo', '42_firm_size']:
    importlib.import_module(f"{__name__}.n{_name}")
