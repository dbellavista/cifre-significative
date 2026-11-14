"""Indicator catalog: one entry per series pulled from Eurostat.

`filters` are Eurostat dimension filters (name -> code). Every dimension not
listed must collapse to a single value after filtering, otherwise the cleaner
raises (guards against silently mixing units/sectors).
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Indicator:
    name: str
    theme: str
    dataset: str
    filters: dict[str, str | list[str]] = field(default_factory=dict)
    note: str = ""
    keep: tuple[str, ...] = ()  # extra dimensions kept as columns when parsing (multi-dimension tables)
    freq: str = "A"  # "A" annual, "M" monthly, "Q" quarterly: sets the first period requested from the API


EUROSTAT: list[Indicator] = [
    # Output
    Indicator("gdp_real_pc", "output", "nama_10_pc", {"na_item": "B1GQ", "unit": "CLV20_EUR_HAB"}, "chain-linked 2020 EUR per head"),
    Indicator("gdp_growth", "output", "nama_10_gdp", {"na_item": "B1GQ", "unit": "CLV_PCH_PRE"}, "real growth %"),
    Indicator("gdp_pps_pc_eu27", "output", "nama_10_pc", {"na_item": "B1GQ", "unit": "PC_EU27_2020_HAB_MEUR_CP"}, "index EU27=100"),
    Indicator("gdp_nominal_meur", "output", "nama_10_gdp", {"na_item": "B1GQ", "unit": "CP_MEUR"}, "current prices, EUR million"),
    # Investment
    Indicator("gfcf_pct_gdp", "investment", "nama_10_gdp", {"na_item": "P51G", "unit": "PC_GDP"}),
    Indicator("rd_pct_gdp", "investment", "rd_e_gerdtot", {"sectperf": "TOTAL", "unit": "PC_GDP"}),
    # Debt / fiscal
    Indicator("gov_debt_pct_gdp", "fiscal", "gov_10dd_edpt1", {"unit": "PC_GDP", "sector": "S13", "na_item": "GD"}),
    Indicator("gov_balance_pct_gdp", "fiscal", "gov_10dd_edpt1", {"unit": "PC_GDP", "sector": "S13", "na_item": "B9"}),
    Indicator("gov_interest_pct_gdp", "fiscal", "gov_10a_main", {"unit": "PC_GDP", "sector": "S13", "na_item": "D41PAY"}),
    Indicator("gov_pubinv_pct_gdp", "investment", "gov_10a_main", {"unit": "PC_GDP", "sector": "S13", "na_item": "P51G"}),
    # Demography
    Indicator("population", "demography", "demo_pjan", {"age": "TOTAL", "sex": "T", "unit": "NR"}, "1 Jan"),
    Indicator("fertility_rate", "demography", "demo_find", {"indic_de": "TOTFERRT"}),
    Indicator("old_age_dependency", "demography", "demo_pjanind", {"indic_de": "OLDDEP1"}, "65+/15-64 %"),
    Indicator("median_age", "demography", "demo_pjanind", {"indic_de": "MEDAGEPOP"}),
    # Labour
    Indicator("unemployment_rate", "labour", "lfsa_urgan", {"age": "Y15-74", "sex": "T", "unit": "PC", "citizen": "TOTAL"}),
    Indicator("youth_unemployment_rate", "labour", "lfsa_urgan", {"age": "Y15-24", "sex": "T", "unit": "PC", "citizen": "TOTAL"}),
    Indicator("employment_rate_20_64", "labour", "lfsa_ergan", {"age": "Y20-64", "sex": "T", "unit": "PC", "citizen": "TOTAL"}),
    Indicator("employment_rate_20_64_female", "labour", "lfsa_ergan", {"age": "Y20-64", "sex": "F", "unit": "PC", "citizen": "TOTAL"}),
    Indicator("employment_rate_55_64", "labour", "lfsa_ergan", {"age": "Y55-64", "sex": "T", "unit": "PC", "citizen": "TOTAL"}),
    Indicator("neet_rate_15_29", "labour", "edat_lfse_20", {"age": "Y15-29", "sex": "T", "unit": "PC", "wstatus": "NEMP"}),
    Indicator("labour_productivity_pps_eu27", "labour", "nama_10_lp_ulc", {"na_item": "RLPR_HW", "unit": "I20"}, "real labour productivity per hour, index 2020=100 (verify)"),
    Indicator("nominal_ulc", "labour", "nama_10_lp_ulc", {"na_item": "NULC_HW", "unit": "I20"}, "index 2020=100"),
    # Education
    Indicator("tertiary_attainment_25_34", "education", "edat_lfse_03", {"age": "Y25-34", "sex": "T", "isced11": "ED5-8", "unit": "PC"}),
    Indicator("early_leavers_18_24", "education", "edat_lfse_14", {"age": "Y18-24", "sex": "T", "unit": "PC", "wstatus": "POP"}),
    Indicator("edu_spend_pct_gdp", "education", "gov_10a_exp", {"unit": "PC_GDP", "sector": "S13", "cofog99": "GF09", "na_item": "TE"}),
    # Pensions / social
    Indicator("pension_spend_pct_gdp", "pensions", "gov_10a_exp", {"unit": "PC_GDP", "sector": "S13", "cofog99": "GF1002", "na_item": "TE"}, "COFOG old age"),
    Indicator("arop_65plus", "pensions", "ilc_li02", {"age": "Y_GE65", "sex": "T", "unit": "PC", "statinfo": "MED_EI", "rskpovth": "B_60"}, "60% of median"),
    Indicator("gini", "inequality", "ilc_di12", {"age": "TOTAL", "statinfo": "GINI_HND"}),
    Indicator("arope", "inequality", "ilc_peps01n", {"age": "TOTAL", "sex": "T", "unit": "PC"}, "2015+ (new definition)"),
    Indicator("arope_old", "inequality", "ilc_peps01", {"age": "TOTAL", "sex": "T", "unit": "PC"}, "2003-2020, methodology break vs arope"),
    # Prices / external
    Indicator("hicp_inflation", "prices", "prc_hicp_aind", {"coicop": "CP00", "unit": "RCH_A_AVG"}),
    Indicator("current_account_pct_gdp", "external", "tipsbp20", {"unit": "PC_GDP", "partner": "WRL_REST"}),
    Indicator("long_term_yield", "prices", "irt_lt_mcby_a", {"int_rt": "MCBY"}, "10y maastricht criterion bond yield"),
    # Wages and gender equality
    Indicator("gender_pay_gap", "gender", "sdg_05_20", {"unit": "PC", "nace_r2": "B-S_X_O"}, "unadjusted, hourly gross earnings, firms 10+, % of male pay"),
    Indicator("gender_pension_gap_65", "gender", "ilc_pnp13", {"age": "Y_GE65", "unit": "PC", "statinfo": "AVG"}, "2015+; % by which women's pensions are lower"),
    Indicator("employment_rate_20_64_male", "gender", "lfsa_ergan", {"age": "Y20-64", "sex": "M", "unit": "PC", "citizen": "TOTAL"}),
    Indicator("part_time_female_20_64", "gender", "lfsa_eppga", {"unit": "PC", "sex": "F", "age": "Y20-64"}, "% of employed women"),
    Indicator("part_time_male_20_64", "gender", "lfsa_eppga", {"unit": "PC", "sex": "M", "age": "Y20-64"}, "% of employed men"),
    Indicator("inactive_care_female_25_54", "gender", "lfsa_igar", {"unit": "PC_POP", "sex": "F", "age": "Y25-54", "reason": "FAM_CARE"}, "% of women 25-54 outside labour force for care"),
    Indicator("inactive_care_male_25_54", "gender", "lfsa_igar", {"unit": "PC_POP", "sex": "M", "age": "Y25-54", "reason": "FAM_CARE"}),
    Indicator("managers_total_ths", "gender", "lfsa_egais", {"unit": "THS_PER", "wstatus": "EMP", "age": "Y15-64", "isco08": "OC1", "sex": "T"}, "helper for women_managers_share_pct"),
    Indicator("managers_female_ths", "gender", "lfsa_egais", {"unit": "THS_PER", "wstatus": "EMP", "age": "Y15-64", "isco08": "OC1", "sex": "F"}, "helper for women_managers_share_pct"),
    Indicator("net_earnings_pps", "wages", "earn_nt_net", {"currency": "PPS", "estruct": "NET", "ecase": "P1_NCH_AW100"}, "annual net earnings, single no children, average worker, PPS"),
    Indicator("gross_earnings_pps", "wages", "earn_nt_net", {"currency": "PPS", "estruct": "GRS", "ecase": "P1_NCH_AW100"}, "annual gross earnings, same case, PPS"),
    Indicator("tax_wedge_low_wage", "wages", "earn_nt_taxwedge", {"unit": "RT"}, "tax wedge on labour costs, low-wage earners (%)"),
    Indicator("labour_cost_hourly_eur", "wages", "lc_lci_lev", {"unit": "EUR", "lcstruct": "D1_D4_MD5", "nace_r2": "B-S_X_O"}, "hourly labour cost; biennial before 2020"),
    Indicator("gender_overall_earnings_gap", "gender", "earn_gr_goeg", {"age": "Y15-64", "nace_r2": "B-S_X_O", "unit": "PC"}, "4-yearly (2010, 2014, 2018, 2022); combines hourly pay, hours and employment"),
    Indicator("involuntary_part_time_female_15_64", "gender", "lfsa_eppgai", {"unit": "PC", "sex": "F", "age": "Y15-64"}, "% of part-time employed women who want full time"),
    Indicator("involuntary_part_time_male_15_64", "gender", "lfsa_eppgai", {"unit": "PC", "sex": "M", "age": "Y15-64"}),
    # Higher education
    Indicator("tertiary_attainment_25_34_female", "education", "edat_lfse_03", {"age": "Y25-34", "sex": "F", "isced11": "ED5-8", "unit": "PC"}),
    Indicator("tertiary_attainment_25_34_male", "education", "edat_lfse_03", {"age": "Y25-34", "sex": "M", "isced11": "ED5-8", "unit": "PC"}),
    Indicator("tertiary_attainment_25_64", "education", "edat_lfse_03", {"age": "Y25-64", "sex": "T", "isced11": "ED5-8", "unit": "PC"}),
    Indicator("tertiary_spend_per_student_pps", "education", "educ_uoe_fine09", {"unit": "PPS", "isced11": "ED5-8"}, "public expenditure per tertiary student, PPS"),
    Indicator("recent_grads_employment_rate", "education", "edat_lfse_24", {"age": "Y20-34", "sex": "T", "isced11": "ED5-8", "duration": "Y_LE3", "unit": "PC"}, "tertiary graduates 1-3 years after finishing"),
    Indicator("recent_grads_employment_rate_female", "education", "edat_lfse_24", {"age": "Y20-34", "sex": "F", "isced11": "ED5-8", "duration": "Y_LE3", "unit": "PC"}),
    Indicator("recent_grads_employment_rate_male", "education", "edat_lfse_24", {"age": "Y20-34", "sex": "M", "isced11": "ED5-8", "duration": "Y_LE3", "unit": "PC"}),
    Indicator("recent_secondary_grads_employment_rate", "education", "edat_lfse_24", {"age": "Y20-34", "sex": "T", "isced11": "ED3_4", "duration": "Y_LE3", "unit": "PC"}, "upper secondary/post-secondary graduates, comparator"),
    # Health outcomes, childcare, older workers
    Indicator("life_expectancy_birth", "health", "demo_mlexpec", {"unit": "YR", "sex": "T", "age": "Y_LT1"}),
    Indicator("healthy_life_years_birth", "health", "hlth_hlye", {"unit": "YR", "sex": "T", "hlth_hle": "HLY_Y0"}),
    Indicator("healthy_life_years_65", "health", "hlth_hlye", {"unit": "YR", "sex": "T", "hlth_hle": "HLY_Y65"}),
    Indicator("treatable_mortality_rate", "health", "hlth_cd_apr", {"mortalit": "TRT", "sex": "T", "icd10": "TOTAL", "unit": "RT"}, "deaths per 100,000 that timely and effective care could have avoided"),
    Indicator("preventable_mortality_rate", "health", "hlth_cd_apr", {"mortalit": "PRVT", "sex": "T", "icd10": "TOTAL", "unit": "RT"}, "deaths per 100,000 preventable by public health and prevention"),
    Indicator("childcare_under3_30h_plus_pct", "gender", "ilc_caindformal", {"age": "Y_LT3", "duration": "H_GE30", "unit": "PC"}, "children under 3 in formal care 30+ hours/week"),
    Indicator("childcare_under3_1_29h_pct", "gender", "ilc_caindformal", {"age": "Y_LT3", "duration": "H1-29", "unit": "PC"}, "children under 3 in formal care 1-29 hours/week"),
    Indicator("ece_participation_3_to_school_pct", "education", "educ_uoe_enra21", {"sex": "T", "unit": "PC"}, "children from age 3 to compulsory school age in early childhood education"),
    Indicator("employment_rate_60_64", "labour", "lfsa_ergan", {"age": "Y60-64", "sex": "T", "unit": "PC", "citizen": "TOTAL"}),
    Indicator("employment_rate_60_64_female", "labour", "lfsa_ergan", {"age": "Y60-64", "sex": "F", "unit": "PC", "citizen": "TOTAL"}),
    Indicator("employment_rate_60_64_male", "labour", "lfsa_ergan", {"age": "Y60-64", "sex": "M", "unit": "PC", "citizen": "TOTAL"}),
    # Crime (police-recorded: definitions and reporting practices differ across countries, compare trends more than levels)
    Indicator("homicide_rate_police", "crime", "crim_off_cat", {"iccs": "ICCS0101", "unit": "P_HTHAB"}, "police-recorded intentional homicide per 100,000"),
    Indicator("sexual_violence_rate_police", "crime", "crim_off_cat", {"iccs": "ICCS0301", "unit": "P_HTHAB"}, "police-recorded sexual violence per 100,000"),
    Indicator("robbery_rate_police", "crime", "crim_off_cat", {"iccs": "ICCS0401", "unit": "P_HTHAB"}, "police-recorded robbery per 100,000"),
    Indicator("burglary_residential_rate_police", "crime", "crim_off_cat", {"iccs": "ICCS05012", "unit": "P_HTHAB"}, "police-recorded burglary of private homes per 100,000 (Italy does not report total burglary)"),
    Indicator("theft_rate_police", "crime", "crim_off_cat", {"iccs": "ICCS0502", "unit": "P_HTHAB"}, "police-recorded theft per 100,000"),
    Indicator("drug_offences_rate_police", "crime", "crim_off_cat", {"iccs": "ICCS0601", "unit": "P_HTHAB"}, "police-recorded drug offences per 100,000"),
    Indicator("fraud_rate_police", "crime", "crim_off_cat", {"iccs": "ICCS0701", "unit": "P_HTHAB"}, "police-recorded fraud per 100,000"),
    Indicator("homicide_victims_per_100k", "crime", "crim_hom_soff", {"iccs": "ICCS0101", "leg_stat": "PER_VICT", "sex": "T", "unit": "P_HTHAB"}),
    Indicator("homicide_victims_female_per_100k", "crime", "crim_hom_soff", {"iccs": "ICCS0101", "leg_stat": "PER_VICT", "sex": "F", "unit": "P_HTHAB"}),
    Indicator("prison_held_nr", "crime", "crim_pris_cap", {"indic_cr": "PRIS_ACT_CAP", "unit": "NR"}, "helper: persons held"),
    Indicator("prison_capacity_nr", "crime", "crim_pris_cap", {"indic_cr": "PRIS_OFF_CAP", "unit": "NR"}, "helper: official prison capacity"),
    Indicator("prison_population_per_100k", "crime", "crim_pris_cap", {"indic_cr": "PRIS_ACT_CAP", "unit": "P_HTHAB"}),
    Indicator("crime_perception_pct", "crime", "ilc_mddw03", {"hhcomp": "TOTAL", "rskpovth": "TOTAL", "unit": "PC"}, "% of population reporting crime, violence or vandalism in the area"),
    # Migration flows and stocks (administrative registration: emigration is under-recorded in several countries, Italy among them)
    Indicator("immigration_total_nr", "migration", "migr_imm8", {"agedef": "COMPLET", "age": "TOTAL", "unit": "NR", "sex": "T"}, "immigrants in the year (incl. returning citizens)"),
    Indicator("emigration_total_nr", "migration", "migr_emi2", {"agedef": "COMPLET", "age": "TOTAL", "unit": "NR", "sex": "T"}, "emigrants in the year (as registered by the origin country)"),
    Indicator("net_migration_rate_per_1000", "migration", "demo_gind", {"indic_de": "CNMIGRATRT"}, "crude net migration rate incl. statistical adjustment"),
    Indicator("asylum_first_applicants_nr", "migration", "migr_asyappctza", {"citizen": "TOTAL", "applicant": "FRST", "sex": "T", "age": "TOTAL", "unit": "PER"}, "first-time asylum applicants"),
    Indicator("first_permits_nr", "migration", "migr_resfirst", {"reason": "TOTAL", "citizen": "TOTAL", "duration": "TOTAL", "unit": "PER"}, "first residence permits issued (non-EU citizens)"),
    Indicator("first_permits_work_nr", "migration", "migr_resfirst", {"reason": "EMP", "citizen": "TOTAL", "duration": "TOTAL", "unit": "PER"}, "first residence permits for employment"),
    Indicator("inwork_poverty_18_64", "inequality", "ilc_iw01", {"wstatus": "EMP", "sex": "T", "age": "Y18-64", "unit": "PC"}, "in-work at-risk-of-poverty rate"),
]

# Monthly (m_) and quarterly (q_) series: seasonally (and calendar) adjusted where Eurostat offers it -> data/processed/highfreq.parquet.
EUROSTAT_HIGHFREQ: list[Indicator] = [
    Indicator("m_unemployment_rate", "labour", "une_rt_m", {"s_adj": "SA", "age": "TOTAL", "unit": "PC_ACT", "sex": "T"}, "% of active population, seasonally adjusted", freq="M"),
    Indicator("m_youth_unemployment_rate", "labour", "une_rt_m", {"s_adj": "SA", "age": "Y_LT25", "unit": "PC_ACT", "sex": "T"}, "under 25, seasonally adjusted", freq="M"),
    Indicator("m_hicp_inflation", "prices", "prc_hicp_minr", {"unit": "RCH_A", "coicop18": "TOTAL"}, "all-items HICP (ECOICOP ver.2 dataset, full history; the old prc_hicp_manr stops in 2025-12), annual rate of change", freq="M"),
    Indicator("m_hicp_energy_inflation", "prices", "prc_hicp_minr", {"unit": "RCH_A", "coicop18": "NRG"}, "HICP energy, annual rate of change", freq="M"),
    Indicator("m_hicp_core_inflation", "prices", "prc_hicp_minr", {"unit": "RCH_A", "coicop18": "TOT_X_NRG_FOOD_NP"}, "HICP excluding energy, food, alcohol and tobacco, annual rate", freq="M"),
    Indicator("m_hicp_services_inflation", "prices", "prc_hicp_minr", {"unit": "RCH_A", "coicop18": "SERV"}, "HICP services, annual rate of change (a deflator for the private-services agreement groups of notebook 39)", freq="M"),
    Indicator("m_hicp_goods_inflation", "prices", "prc_hicp_minr", {"unit": "RCH_A", "coicop18": "GD"}, "HICP goods (food, energy, industrial goods), annual rate of change (a deflator for the industry, agriculture and construction agreement groups of notebook 39)", freq="M"),
    Indicator("m_industrial_production", "output", "sts_inpr_m", {"indic_bt": "PRD", "nace_r2": "B-D", "s_adj": "SCA", "unit": "I21"}, "industry volume index, 2021 = 100", freq="M"),
    Indicator("m_retail_volume", "output", "sts_trtu_m", {"indic_bt": "VOL_SLS", "nace_r2": "G47", "s_adj": "SCA", "unit": "I21"}, "retail trade volume index, 2021 = 100", freq="M"),
    Indicator("m_long_term_yield", "prices", "irt_lt_mcby_m", {"int_rt": "MCBY"}, "10-year government bond yield (EMU convergence criterion), %", freq="M"),
    Indicator("m_esi", "sentiment", "ei_bssi_m_r2", {"indic": "BS-ESI-I", "s_adj": "SA"}, "economic sentiment indicator, long-run mean 100", freq="M"),
    Indicator("m_consumer_confidence", "sentiment", "ei_bssi_m_r2", {"indic": "BS-CSMCI-BAL", "s_adj": "SA"}, "consumer confidence balance", freq="M"),
    Indicator("m_industry_confidence", "sentiment", "ei_bssi_m_r2", {"indic": "BS-ICI-BAL", "s_adj": "SA"}, "industry confidence balance", freq="M"),
    Indicator("m_services_confidence", "sentiment", "ei_bssi_m_r2", {"indic": "BS-SCI-BAL", "s_adj": "SA"}, "services confidence balance", freq="M"),
    Indicator("q_gdp_growth", "output", "namq_10_gdp", {"unit": "CLV_PCH_PRE", "s_adj": "SCA", "na_item": "B1GQ"}, "real GDP, % change on previous quarter", freq="Q"),
    Indicator("q_employment_ths", "labour", "namq_10_a10_e", {"unit": "THS_PER", "s_adj": "SCA", "nace_r2": "TOTAL", "na_item": "EMP_DC"}, "employed persons (domestic concept), thousands", freq="Q"),
    Indicator("q_hours_ths", "labour", "namq_10_a10_e", {"unit": "THS_HW", "s_adj": "SCA", "nace_r2": "TOTAL", "na_item": "EMP_DC"}, "hours worked, thousands", freq="Q"),
    Indicator("q_labour_cost_yoy", "wages", "lc_lci_r2_q", {"s_adj": "CA", "unit": "PCH_SM", "nace_r2": "B-S", "lcstruct": "D1_D4_MD5"}, "hourly labour cost index, % change on same quarter a year earlier", freq="Q"),
]

# NUTS2 (Italy regional) fetched separately in the regional step.

# Regional (NUTS2 and coarser) series; fetched for all geos and filtered at build time.
EUROSTAT_REGIONAL: list[Indicator] = [
    Indicator("nuts2_gdp_pps_pc_eu27", "output", "nama_10r_2gdp", {"unit": "PPS_EU27_2020_HAB"}, "PPS per inhabitant (EUR), not an index"),
    Indicator("nuts2_unemployment_rate", "labour", "lfst_r_lfu3rt", {"sex": "T", "age": "Y15-74", "unit": "PC", "isced11": "TOTAL"}),
    Indicator("nuts2_tertiary_attainment_25_64", "education", "edat_lfse_04", {"sex": "T", "age": "Y25-64", "isced11": "ED5-8", "unit": "PC"}),
    Indicator("nuts2_employment_rate_20_64_female", "labour", "lfst_r_lfe2emprt", {"sex": "F", "age": "Y20-64", "unit": "PC"}),
    Indicator("nuts2_gdp_mio_eur", "output", "nama_10r_2gdp", {"unit": "MIO_EUR"}, "GDP at current prices, EUR million"),
    Indicator("nuts2_hours_ths", "labour", "nama_10r_2emhrw", {"unit": "THS", "wstatus": "EMP", "nace_r2": "TOTAL"}, "hours worked, thousands (all employed)"),
    Indicator("nuts2_population", "demography", "demo_r_d2jan", {"sex": "T", "age": "TOTAL", "unit": "NR"}),
    # Labour and skills
    Indicator("nuts2_employment_rate_20_64", "labour", "lfst_r_lfe2emprt", {"sex": "T", "age": "Y20-64", "unit": "PC"}),
    Indicator("nuts2_employment_rate_20_64_male", "labour", "lfst_r_lfe2emprt", {"sex": "M", "age": "Y20-64", "unit": "PC"}),
    Indicator("nuts2_employment_rate_55_64", "labour", "lfst_r_lfe2emprt", {"sex": "T", "age": "Y55-64", "unit": "PC"}),
    Indicator("nuts2_youth_unemployment_rate", "labour", "lfst_r_lfu3rt", {"sex": "T", "age": "Y15-24", "unit": "PC", "isced11": "TOTAL"}, "unemployment rate 15-24"),
    Indicator("nuts2_long_term_unemployment_pct_active", "labour", "lfst_r_lfu2ltu", {"sex": "T", "age": "Y15-74", "unit": "PC_ACT", "isced11": "TOTAL"}, "long-term unemployed (12+ months), % of active population"),
    Indicator("nuts2_neet_rate_15_29", "labour", "edat_lfse_22", {"sex": "T", "age": "Y15-29"}),
    Indicator("nuts2_early_leavers_18_24", "education", "edat_lfse_16", {"sex": "T", "age": "Y18-24", "unit": "PC"}),
    # Innovation and investment
    Indicator("nuts2_rd_pct_gdp", "investment", "rd_e_gerdreg", {"sectperf": "TOTAL", "unit": "PC_GDP"}, "R&D expenditure, % of regional GDP"),
    Indicator("nuts2_rd_personnel_pct_employment", "investment", "rd_p_persreg", {"prof_pos": "TOTAL", "sex": "T", "sectperf": "TOTAL", "unit": "PC_EMP_FTE"}, "R&D personnel (FTE), % of total employment"),
    Indicator("nuts2_gfcf_mio_eur", "investment", "nama_10r_2gfcf", {"sector": "S1", "currency": "MIO_EUR", "nace_r2": "TOTAL"}, "gross fixed capital formation, EUR million"),
    Indicator("nuts2_coe_mio_eur", "wages", "nama_10r_2coe", {"currency": "MIO_EUR", "nace_r2": "TOTAL"}, "compensation of employees, EUR million"),
    Indicator("nuts2_broadband_households_pct", "investment", "isoc_r_broad_h", {"unit": "PC_HH"}, "households with broadband access"),
    # Demography and health
    Indicator("nuts2_fertility_rate", "demography", "demo_r_find2", {"indic_de": "TOTFERRT", "unit": "NR"}),
    Indicator("nuts2_life_expectancy_birth", "health", "demo_r_mlifexp", {"sex": "T", "age": "Y_LT1", "unit": "YR"}),
    Indicator("nuts2_old_age_dependency", "demography", "demo_r_pjanind2", {"indic_de": "OLDDEP1", "unit": "PC"}, "population 65+ per 100 aged 15-64"),
    Indicator("nuts2_median_age", "demography", "demo_r_pjanind2", {"indic_de": "MEDAGEPOP", "unit": "YR"}),
    Indicator("nuts2_net_migration_rate", "migration", "demo_r_gind3", {"indic_de": "CNMIGRATRT"}, "crude rate of net migration plus statistical adjustment, per 1000"),
    Indicator("nuts2_physicians_per_100k", "health", "hlth_rs_prsrg", {"isco08": "OC221", "unit": "P_HTHAB"}, "medical doctors per 100,000 inhabitants; series ends 2021"),
    # Poverty
    Indicator("nuts2_arope", "inequality", "ilc_peps11n", {"unit": "PC"}, "at risk of poverty or social exclusion; coverage starts later"),
]

# Structural business statistics by NUTS2 (2008-2020; there is no all-activity total, so the sections B-N without K are kept and combined in `build.build_regional_sbs`)
SBS_SECTIONS = ["B", "C", "D", "E", "F", "G", "H", "I", "J", "L", "M", "N"]
EUROSTAT_REGIONAL_SBS: list[Indicator] = [
    Indicator("nuts2_sbs_local_units", "structure", "sbs_r_nuts06_r2", {"indic_sb": "V11210", "nace_r2": SBS_SECTIONS}, "local units, number", keep=("nace_r2",)),
    Indicator("nuts2_sbs_persons_employed", "structure", "sbs_r_nuts06_r2", {"indic_sb": "V16110", "nace_r2": SBS_SECTIONS}, "persons employed, number", keep=("nace_r2",)),
    Indicator("nuts2_sbs_wages_meur", "structure", "sbs_r_nuts06_r2", {"indic_sb": "V13320", "nace_r2": SBS_SECTIONS}, "wages and salaries, EUR million", keep=("nace_r2",)),
]

# Industry detail (mutually exclusive NACE sections, A*10-style): kept as extra dimension `nace_r2`.
SECTORS = ["A", "B-E", "F", "G-I", "J", "K", "L", "M_N", "O-Q", "R-U"]
EUROSTAT_SECTORS: list[Indicator] = [
    Indicator("sector_gva_clv20_meur", "output", "nama_10_a64", {"na_item": "B1G", "unit": "CLV20_MEUR", "nace_r2": SECTORS}, "real GVA, chain-linked 2020 EUR m"),
    Indicator("sector_gva_cp_meur", "output", "nama_10_a64", {"na_item": "B1G", "unit": "CP_MEUR", "nace_r2": SECTORS}, "nominal GVA, EUR m"),
    Indicator("sector_hours_ths", "labour", "nama_10_a64_e", {"na_item": "EMP_DC", "unit": "THS_HW", "nace_r2": SECTORS}, "hours worked, thousands"),
]

# Structural business statistics by employment size class -> data/processed/size_class.parquet (the economy-wide row is the published aggregate of each vintage: sums of sections lose most countries to confidential cells). Two vintages that are never spliced: `sbs_sc_sca_r2` (2005-2020, value added at factor cost) and
# `sbs_sc_ovw` (2021 onwards, new regulation, value added at basic prices). Only the five disjoint classes are kept (0-9, 10-19, 20-49, 50-249, 250+) plus TOTAL as a check; 2-9 and 0-1 of the new set overlap them.
SIZE_NACE = ["B-N_S95_X_K"] + SBS_SECTIONS  # old set: business economy without K, repairs S95 included
SIZE_NACE_NEW = ["B-S_X_O_S94"] + SBS_SECTIONS  # new set: industry, construction and market services (wider scope than B-N)
SIZE_OLD = ["TOTAL", "0-9", "10-19", "20-49", "50-249", "GE250"]
SIZE_NEW = ["TOTAL", "0-9", "10-19", "20-49", "50-249", "GE250"]
EUROSTAT_SIZE: list[Indicator] = [
    Indicator("size_old_enterprises", "structure", "sbs_sc_sca_r2", {"indic_sb": "V11110", "nace_r2": SIZE_NACE, "size_emp": SIZE_OLD}, "enterprises, number, 2005-2020", keep=("nace_r2", "size_emp")),
    Indicator("size_old_persons", "structure", "sbs_sc_sca_r2", {"indic_sb": "V16110", "nace_r2": SIZE_NACE, "size_emp": SIZE_OLD}, "persons employed, number, 2005-2020", keep=("nace_r2", "size_emp")),
    Indicator("size_old_va_meur", "structure", "sbs_sc_sca_r2", {"indic_sb": "V12150", "nace_r2": SIZE_NACE, "size_emp": SIZE_OLD}, "value added at factor cost, EUR million, 2005-2020", keep=("nace_r2", "size_emp")),
    Indicator("size_new_enterprises", "structure", "sbs_sc_ovw", {"indic_sbs": "ENT_NR", "nace_r2": SIZE_NACE_NEW, "size_emp": SIZE_NEW}, "enterprises, number, 2021 onwards", keep=("nace_r2", "size_emp")),
    Indicator("size_new_persons", "structure", "sbs_sc_ovw", {"indic_sbs": "EMP_NR", "nace_r2": SIZE_NACE_NEW, "size_emp": SIZE_NEW}, "persons employed, number, 2021 onwards", keep=("nace_r2", "size_emp")),
    Indicator("size_new_va_meur", "structure", "sbs_sc_ovw", {"indic_sbs": "AV_MEUR", "nace_r2": SIZE_NACE_NEW, "size_emp": SIZE_NEW}, "value added, EUR million, 2021 onwards (new regulation, not comparable with the old set)", keep=("nace_r2", "size_emp")),
    Indicator("size_price_level", "prices", "prc_ppp_ind", {"na_item": "PLI_EU27_2020", "ppp_cat": "GDP"}, "GDP price level index, EU27 = 100: puts value added per person on a common scale (a GDP-wide level, an approximation for sector value added)"),
]

# Government spending detail (kept extra dimensions): by function (COFOG) and by economic nature. -> spending.parquet
COFOG_ITEMS = ["TOTAL", "GF01", "GF0107", "GF02", "GF03", "GF04", "GF0401", "GF0405", "GF05", "GF06", "GF07", "GF08", "GF09", "GF10",
               "GF1001", "GF1002", "GF1003", "GF1004", "GF1005", "GF1006", "GF1007", "GF1008", "GF1009"]
NATURE_ITEMS = ["TE", "TR", "D1PAY", "P2", "D62_D632PAY", "D41PAY", "D3PAY", "P51G", "D9PAY", "D92PAY", "D99PAY", "D7PAY", "D2REC", "D5REC", "D61REC"]
EUROSTAT_SPENDING: list[Indicator] = [
    Indicator("spend_cofog", "fiscal", "gov_10a_exp", {"unit": ["PC_GDP", "PC_TOT", "MIO_EUR"], "sector": "S13", "na_item": "TE", "cofog99": COFOG_ITEMS}, "general government expenditure by function"),
    Indicator("spend_nature", "fiscal", "gov_10a_main", {"unit": "PC_GDP", "sector": "S13", "na_item": NATURE_ITEMS}, "expenditure/revenue by economic nature, % of GDP"),
]

# (indicator, year) points dropped at build time because the source value is a known anomaly.
# Empty since 2026-11-10: Eurostat republished earn_nt_net with 2024 on the annual scale (it was monthly, about 1/12 of the neighbouring years).
# The 2024 break flag stays: from 2024 the average wage has a new basis (gross euro pay of the average worker: Italy +13%, Germany -7% on 2023), so 2024 onwards is not comparable with 2023 and before (SERIES_BREAKS).
EXCLUDED_POINTS: dict[tuple[str, int], str] = {}
# indicator -> first year of a new basis that is not comparable with the years before (never compare or average across it)
SERIES_BREAKS = {"net_earnings_pps": 2024, "gross_earnings_pps": 2024, "tax_wedge_low_wage": 2024}  # the three series of the earn_nt tables in the panel

# Higher-education detail tables (extra dimensions kept) -> data/processed/highered.parquet, plus derived rates in the panel
_FIELDS = ["TOTAL", "F00", "F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10"]
_LEVELS = ["ED5-8", "ED5", "ED6", "ED7", "ED8"]  # all tertiary, short-cycle, bachelor, master, doctoral
_AGES = (["TOTAL"] + [f"Y{a}" for a in range(20, 50)] + ["Y50-54", "Y55-59", "Y60-64", "Y_GE65"])
EUROSTAT_HIGHERED: list[Indicator] = [
    Indicator("he_grads_field", "education", "educ_uoe_grad02", {"unit": "NR", "isced11": _LEVELS, "iscedf13": _FIELDS, "sex": ["T", "M", "F"]}, "graduates by level, field, sex", keep=("isced11", "iscedf13", "sex")),
    Indicator("he_students_field", "education", "educ_uoe_enrt03", {"unit": "NR", "isced11": _LEVELS, "iscedf13": _FIELDS, "sex": ["T", "M", "F"]}, "students by level, field, sex", keep=("isced11", "iscedf13", "sex")),
    Indicator("he_grads_age", "education", "educ_uoe_grad01", {"unit": "NR", "isced11": ["ED5-8", "ED6", "ED7", "ED8"], "sex": ["T", "F", "M"], "age": _AGES}, "graduates by level, sex, single year of age", keep=("isced11", "sex", "age")),
    Indicator("he_pop_age", "demography", "demo_pjangroup", {"unit": "NR", "sex": "T", "age": ["Y20-24", "Y25-29", "Y30-34"]}, "population by age group (denominators)", keep=("age",)),
]

# Eurostat EUROPOP2023 projections (demographic indicators by scenario) -> data/processed/projections_eurostat.parquet
EUROSTAT_PROJECTIONS: list[Indicator] = [
    Indicator("proj_demo", "demography", "proj_23ndbi", {"indic_de": ["JAN", "OLDDEP1", "DEPRATIO3", "MEDAGEPOP", "NMIGRAT", "LBIRTH", "DEATH"], "projection": ["BSL", "LFRT", "HMIGR", "NMIGR"]},
              "EUROPOP2023 baseline and variants: population on 1 Jan, old-age dependency, median age, births, deaths, net migration", keep=("indic_de", "projection")),
]

# Finer industries (NACE A*64 leaf codes) -> data/processed/sectors64.parquet
A64 = ["A01", "A02", "A03", "B", "C10-C12", "C13-C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30", "C31_C32", "C33",
       "D", "E36", "E37-E39", "F", "G45", "G46", "G47", "H49", "H50", "H51", "H52", "H53", "I", "J58", "J59_J60", "J61", "J62_J63", "K64", "K65", "K66", "L68", "M69_M70", "M71",
       "M72", "M73", "M74_M75", "N77", "N78", "N79", "N80-N82", "O84", "P85", "Q86", "Q87_Q88", "R90-R92", "R93", "S94", "S95", "S96"]
EUROSTAT_SECTORS64: list[Indicator] = [
    Indicator("sector64_gva_clv20_meur", "output", "nama_10_a64", {"na_item": "B1G", "unit": "CLV20_MEUR", "nace_r2": A64}, "real GVA by A*64 industry", keep=("nace_r2",)),
    Indicator("sector64_gva_cp_meur", "output", "nama_10_a64", {"na_item": "B1G", "unit": "CP_MEUR", "nace_r2": A64}, "nominal GVA by A*64 industry", keep=("nace_r2",)),
    Indicator("sector64_hours_ths", "labour", "nama_10_a64_e", {"na_item": "EMP_DC", "unit": "THS_HW", "nace_r2": A64}, "hours worked by A*64 industry", keep=("nace_r2",)),
]

# Annual earnings by household case: gross, tax, social contributions, family benefits, net -> data/processed/earnings_cases.parquet
EUROSTAT_EARNINGS: list[Indicator] = [
    Indicator("earnings_cases", "wages", "earn_nt_net", {"currency": "EUR", "estruct": ["GRS", "TAX", "SOC", "FAM", "NET"], "ecase": ["P1_NCH_AW50", "P1_NCH_AW100", "P1_NCH_AW167", "P1_CH2_AW67", "CPL_CH2_AW100"]},
              "single/couple, with/without children, at 50-167% of the average wage", keep=("estruct", "ecase")),
]

# Income distribution (EU-SILC) -> data/processed/income_dist.parquet. Kept out of the annual panel on purpose: ~10 more correlated inequality series would feed the national screens
EUROSTAT_INCOME: list[Indicator] = [
    Indicator("income_share", "inequality", "ilc_di01", {"statinfo": "SHARE", "unit": "PPS", "quant_inc": ["QU1", "QU2", "QU3", "QU4", "QU5", "D1", "D10"]},
              "share of equivalised disposable income held by each quintile / the bottom and top decile (the share does not depend on the unit)", keep=("quant_inc",)),
    Indicator("s80_s20", "inequality", "ilc_di11", {"age": "TOTAL", "sex": "T", "unit": "RAT"}, "income quintile share ratio S80/S20"),
    Indicator("gini_disp", "inequality", "ilc_di12", {"age": "TOTAL", "statinfo": "GINI_HND"}, "Gini of equivalised disposable income (same series as the panel's `gini`; kept here to pair with the pre-transfer Ginis)"),
    Indicator("gini_pre_pens_incl", "inequality", "ilc_di12b", {}, "Gini of equivalised income before social transfers, pensions INCLUDED in social transfers (pensions removed from income)"),
    Indicator("gini_pre_pens_excl", "inequality", "ilc_di12c", {}, "Gini of equivalised income before social transfers, pensions EXCLUDED from social transfers (pensions stay in market income)"),
]

# Self-employment (EU labour force survey; EU-SILC for poverty by activity status) -> data/processed/selfemp.parquet. Kept out of the annual panel like the income tables
EUROSTAT_SELFEMP: list[Indicator] = [
    Indicator("selfemp_share_status", "labour", "lfsa_esgan", {"unit": "PC_EMP", "citizen": "TOTAL", "age": "Y15-64", "sex": ["T", "M", "F"], "wstatus": ["SELF", "SELF_S", "SELF_NS"]},
              "% of employment aged 15-64: all self-employed, employers (SELF_S) and own-account workers without employees (SELF_NS)", keep=("wstatus", "sex")),
    Indicator("selfemp_share_age", "labour", "lfsa_esgan", {"unit": "PC_EMP", "citizen": "TOTAL", "sex": "T", "wstatus": "SELF", "age": ["Y15-24", "Y25-49", "Y50-64", "Y15-64"]},
              "% of employment in the age band that is self-employed", keep=("age",)),
    Indicator("selfemp_nace_ths", "labour", "lfsa_esgan2", {"unit": "THS_PER", "age": "Y15-64", "sex": "T", "wstatus": "SELF"}, "self-employed persons 15-64 by NACE section, thousands (2008+)", keep=("nace_r2",)),
    Indicator("employed_nace_ths", "labour", "lfsa_egan2", {"unit": "THS_PER", "age": "Y15-64", "sex": "T"}, "employed persons 15-64 by NACE section, thousands (2008+)", keep=("nace_r2",)),
    Indicator("arop_by_status", "inequality", "ilc_li04", {"unit": "PC", "statinfo": "MED_EI", "rskpovth": "B_60", "sex": "T", "age": "Y18-64", "wstatus": ["EMP", "SAL", "NSAL", "UNE"]},
              "at-risk-of-poverty rate (60% of median) by most frequent activity status in the previous year; NSAL = employed persons except employees, i.e. self-employed and family workers", keep=("wstatus",)),
]

# Migration by citizenship and age -> data/processed/migration.parquet (derived per-capita rates go into the panel)
_MIG_AGES = ["TOTAL", "Y20-24", "Y25-29", "Y30-34", "Y35-39"]
EUROSTAT_MIGRATION: list[Indicator] = [
    Indicator("mig_imm_citizen", "migration", "migr_imm1ctz", {"citizen": ["TOTAL", "NAT"], "agedef": "COMPLET", "age": _MIG_AGES, "unit": "NR", "sex": "T"}, "immigrants by citizenship (NAT = nationals, i.e. returns)", keep=("citizen", "age")),
    Indicator("mig_emi_citizen", "migration", "migr_emi1ctz", {"citizen": ["TOTAL", "NAT"], "agedef": "COMPLET", "age": _MIG_AGES, "unit": "NR", "sex": "T"}, "emigrants by citizenship (NAT = nationals leaving)", keep=("citizen", "age")),
    Indicator("mig_pop_citizen", "migration", "migr_pop1ctz", {"citizen": ["TOTAL", "NAT"], "age": "TOTAL", "unit": "NR", "sex": "T"}, "population on 1 January by citizenship", keep=("citizen",)),
]
