# Methodology

How we approach the work in this repository and the statistics behind it. This document explains the *why* and the mathematics; [`analysis_framework.md`](analysis_framework.md) is the per-experiment reference (ids, runners, numbers, limits), [`data_dictionary.md`](data_dictionary.md) describes the tables and [`../reports/findings.md`](../reports/findings.md) holds the results. Function names below point to the code that implements each step.

## 1. Purpose and stance

The question is "where does Italy differ from the other EU27 members, and which of those differences travel together with others?". The project is **descriptive and exploratory**. It does not estimate causal effects. Every claim is an association, a predictive order, or a rank, and carries its limits next to it.

Three facts shape the whole design:

1. **Few units, many variables.** About 27 countries (up to 30 with GBR/USA/JPN), 15-20 years, around 176 annual indicators, about 230 NUTS2 regions. A naive screen of ~15,000 indicator pairs on 27 countries produces hundreds of "significant" correlations by chance.
2. **Italy is one unit.** A country-level claim about Italy rests on one cluster. No test can tell us that Italy is "significantly" different in the usual sense, so we rank Italy among placebo countries instead (section 6).
3. **Dependence everywhere.** Countries share a development factor, time series trend and are autocorrelated, neighbouring regions resemble each other, and many indicators are the same quantity under another name. Each of these inflates apparent evidence if ignored.

The approach follows from this: **pre-specify, correct for multiplicity, test against a null built from the data itself, and report negative results.**

## 2. Principles

| principle | what it means in practice |
|---|---|
| Descriptive, never causal | Wording is "associated with", "precedes", "ranks". The caveat sits in every notebook and dashboard text. Lead-lag results show predictive order only. |
| Italy vs the median of the *other* EU27 | Italy is excluded from its own benchmark (`data.eu_median_ex_italy`). A year is left empty when fewer than 70% of the countries that ever report the series have a value, so a few early reporters cannot pose as "the EU median". |
| Growth, indices and ratios, not euro levels | Different price levels, currencies and base years make levels incomparable. Different source families never share an axis (`wb_gini` vs `gini`). |
| No silent splicing | Series breaks, provisional values and approximations carry flags (`b`, `p`, `backcast`, `linked`) and are dropped from the screens (`corr.BAD_FLAGS`). |
| Pre-specification | A family of tests is fixed before the numbers are seen and the multiplicity correction runs over that family. Choices made after seeing results (for instance the unit of analysis in notebook 39) are declared as such and treated as sensitivity analyses. |
| Calibrated nulls | Every new statistic is run on shuffled or placebo data. The shuffled version must give about zero discoveries, or about 5% of tests below p = 0.05. A procedure that fails its null is fixed or replaced, not reported. |
| Report negatives | "0 of 10 tests survive" is a finding and goes in `reports/findings.md` with its power limit. |
| Takeaways come from executed numbers | Print the numbers, write the text, re-check each quoted figure. Findings go to `reports/findings.md`, limits to `analysis_framework.md`. |

## 3. Data pipeline and what it guarantees

`fetch` downloads raw data into `data/raw/` (cached, with a manifest), `clean.build` produces tidy parquet tables in `data/processed/`. The indicator registry is `catalog.py`; the cleaner (`clean/jsonstat.py`) refuses any dataset with an unfiltered dimension, so each series is one number per country-period.

- **Annual panel** `(iso3, year, indicator, value, flag)`; **regional** NUTS2; **high-frequency** monthly and quarterly; further tables for daily yields, budget laws, collective agreements, tax returns and others.
- **Geography.** NUTS codes change vintage. `clean/nuts.py` maps NUTS 2010, 2013, 2016 regions to 2021 by area overlap with GISCO boundaries: renamed or merged regions map across, split regions are dropped, and no value is ever invented or summed across a split.
- **Flags** are kept as strings (`bp`, `bdu`) and matched by character. Screens drop break and provisional observations; single-window levels drop only unreliable ones.
- **Coverage is checked per series** before use (AROPE from 2015, broadband to 2021, and so on) and recorded in the data dictionary.

## 4. The national correlation screen

Implemented in `analysis/corr.py`, described as a funnel in `analysis_framework.md`. The mathematics of each stage:

### 4.1 Building the feature table

Let $X_{ci}$ be the value of indicator $i$ for country $c$.

- **Level screen:** $X_{ci}$ is the mean over a window (default 2015-19). Averaging damps year-to-year noise and the 2020 shock.
- **Change screen:** window mean minus an earlier window mean (2005-09). This asks whether countries that *moved* together are the same, which removes time-invariant country traits.
- **Within screen:** year-on-year changes demeaned by country *and* by year (two-way demeaning, `within_changes`). Common shocks and fixed country traits drop out; what is left is idiosyncratic co-movement.
- **Income control:** rich countries are better on almost everything, so a raw level screen mostly rediscovers development. `residualise` replaces each indicator by its residual from an OLS fit on log GDP per head, $\tilde x_{ci} = x_{ci} - (\hat a_i + \hat b_i \log y_c)$, and correlates residuals. This asks what moves together *beyond* income. We report both the raw and the income-controlled screen.

### 4.2 Pairwise association

For every admissible pair we compute Spearman's rank correlation $\rho$ on that pair's complete rows (at least $n_{\min}=20$):

$$\rho = \operatorname{corr}\big(\operatorname{rank}(x), \operatorname{rank}(y)\big).$$

Ranks make the statistic robust to outliers and to monotone but non-linear relations, which matters with 27 heterogeneous economies. The p-value is the standard t approximation. The confidence interval uses Fisher's transform

$$z = \operatorname{arctanh}(\rho), \qquad \mathrm{SE} = \frac{1}{\sqrt{n-3}}, \qquad \mathrm{CI} = \tanh(z \pm 1.96\,\mathrm{SE}),$$

which is exact only for Pearson's $r$ and a slight under-estimate of the width for Spearman; it is reported as approximate.

**Excluded pairs** (`is_trivial_pair`, `size_scaled`): pairs that measure the same quantity (same series from two sources, a name that extends the other, same two-token family, a curated duplicate list), size-scaled series (counts, millions of euro, population), and survivors with $|\rho| \ge 0.95$ (almost surely the same quantity with another base year). These filters are heuristic; the top of every list still needs human reading.

### 4.3 Multiple testing: Benjamini-Hochberg

With $m$ tests and ordered p-values $p_{(1)} \le \dots \le p_{(m)}$, the BH adjusted value is

$$q_{(k)} = \min_{j \ge k} \; \min\!\Big(1, \; \frac{m}{j}\,p_{(j)}\Big),$$

and a test is a *discovery* if $q \le 0.05$. This controls the **false discovery rate**, the expected share of false discoveries among the discoveries, under independence or positive dependence. We prefer it to Bonferroni because we want a ranked list of candidates for reading, not zero false positives. `corr.fdr` implements it. $m$ is always the number of pairs actually tested, after exclusions, and for pre-specified hypothesis families it is the size of the family only.

### 4.4 Robustness stamps

Each survivor is re-estimated (`corr.robustness`): leave-one-country-out (every $\rho_{-c}$ keeps the sign and at least 50% of $|\rho|$), without Ireland (multinational profit shifting distorts GDP-type series), and the other estimator (Pearson vs Spearman) agrees in sign. A pair is `robust` only if it passes all three. These checks rarely reject strong survivors, so they are a floor, not a proof.

### 4.5 Split-sample replication

Leave-one-out cannot detect a relation carried by one period. `corr.split_sample` runs the whole funnel separately on 2005-14 and 2015-24 window means; a pair is `replicated` only if it is significant (BH, per half) in both with the same sign. This is a much sharper filter, and it has its own shuffled null (`split_sample_null`).

### 4.6 The shuffled null

`corr.permutation_null` independently permutes each indicator's values across countries, which destroys every cross-indicator association but keeps each marginal distribution, then repeats the screen. A correct procedure yields about zero BH discoveries per run. If it does not, the funnel is anti-conservative and the results are not trusted.

## 5. Panel regressions and predictive order

### 5.1 Fixed-effects panel (`corr.panel_fe`)

For outcome $y_{ct}$ and regressors $x_{ct}$:

$$y_{ct} = \alpha_c + \gamma_t + \beta^\top x_{ct} + \varepsilon_{ct}.$$

Country effects $\alpha_c$ absorb stable traits (institutions, geography, measurement conventions) and year effects $\gamma_t$ absorb shocks common to all countries (2008, 2020). $\beta$ is identified only from within-country deviations relative to the year's cross-country movement. Standard errors are clustered by country (by country prefix for regions) because $\varepsilon_{ct}$ is serially correlated. The model is run with `linearmodels.PanelOLS`.

### 5.2 Lead-lag (E4)

To ask which variable predicts which, we regress $y_t$ on $x_{t-k}$ and $x_t$ on $y_{t-k}$ for $k = 1, 2, 3$ with the same fixed effects. If only one direction is significant we say "x precedes y". This is **predictive precedence, not causation**: a common driver with different lags generates the same pattern, and with entity effects and few years the usual Nickell-type bias applies to dynamic models, which is why we do not include lagged outcomes in the annual models.

### 5.3 Literature hypotheses (E7)

Thirteen hypotheses are written down beforehand from the literature (`docs/literature/`, `experiments.HYPOTHESES`), each mapped to indicators. BH runs across those hypotheses only, with and without Ireland. This is the confirmatory side of the project; the pair screens are the exploratory side, and results from the two are never pooled into one claim.

## 6. Single-country claims: placebo ranking

Italy is one cluster, so the standard error of "Italy differs" cannot be estimated from Italy. Instead we apply the same statistic to *every* country and ask where Italy lands.

For a statistic $T_c$ computed for each of $N$ countries (a gap to the median of the others, a regression slope, a deficit forecast error), the one-sided placebo p-value for Italy is

$$p_{\text{IT}} = \frac{\#\{c : T_c \ge T_{\text{IT}}\}}{N},$$

and the two-sided version uses $|T_c|$. Ties count against Italy. The smallest attainable value is $1/N$ (for $N=27$ it is 0.037), so a ranking cannot reach $q < 0.05$ over many tests, and we say so: with short samples the output is an *ordering*, not a discovery. The same device appears as `corr.italy_fit` (Italy's residual from the line fitted on the other countries, in units of their residual SD, flagged at $|z|>2$), `industry_panel` (Italy's slope among country slopes), `income_dist`, `selfemp`, `hf.italy_rank` and `budget_xc`.

A related device is the **gap series**, Italy minus the median of the others. Gap *levels* trend, so correlating them would be spurious; E5 correlates year-on-year *changes* of gaps. With 14-20 changes the power is low and the test is labelled accordingly.

## 7. Permutation inference for clustered data

Asymptotic p-values assume independent observations. Where that fails we replace them with randomisation tests: shuffle the variable in a way that respects the dependence structure, recompute the statistic, and take

$$p = \frac{1 + \#\{b : |T^{(b)}| \ge |T_{\text{obs}}|\}}{1 + B}.$$

The "+1" keeps the p-value away from zero and makes the test exact in finite samples. What is shuffled is chosen to match the dependence:

- **Regions within countries** (`regional_south.perm_p`, `corr.shuffle_within`). Pooled, country-demeaned Spearman has an asymptotic p that over-rejects a shuffled test (8-9% at a nominal 5%) because countries differ in dispersion, which widens the true null of the pooled $\rho$. Shuffling regions only *within* their country preserves the country structure and restores the nominal 5%.
- **Industries within sectors, institutions across countries, outcomes across years or groups**, depending on the notebook (`perm_pooled`, within-branch shuffles in `contracts`, year shuffles in `budget`). The shuffled unit is always the exchangeable unit under the null.
- **Same-date events count once.** In the daily event tests, events sharing a date are a single test.

For every new family we also report the **calibration of the null**: the share of shuffled tests with $p<0.05$ (target about 5%) and the number of BH discoveries per shuffled family (target about 0).

## 8. Regional (NUTS2) analysis

Regions inside a country share institutions, history and statistics, so 230 regions are far fewer than 230 independent observations (about 27 country clusters).

- **Demeaning.** Each indicator minus its country mean, so we study *within-country* differences; countries with fewer than three regions with data drop out. The log-GDP-per-head fit is then removed from the other variables.
- **Robustness at the cluster level.** Leave-one-out drops whole countries, panel SEs cluster by country, the null shuffles within countries.
- **Mechanical pairs** (GDP per hour vs GDP per head, ratios sharing a numerator or denominator) are never paired.
- **Beta-convergence (R5).** $\Delta \log y_{r} = a_c + \beta \log y_{r,0} + \varepsilon$, with country effects; $\beta<0$ means poorer regions grow faster within their country. SEs are clustered by country, and the richest region per country is dropped as a robustness check.

### 8.1 "Why the South": two designs

Track A (EU-wide) tests a pre-specified family of regional covariates against regional gaps with within-country permutation p-values and BH over the family; Track B (Italy only, 21 regions) is explicitly exploratory, uses permutation within and across macro-zones, and reports leave-one-region-out ranges. Italy's slope is compared with the other countries' by placebo rank (section 6). Putting both side by side shows whether Italy's within-country pattern is the EU pattern or is specific to it.

### 8.2 Spatial dependence (`analysis/spatial.py`)

Neighbouring regions are correlated, which makes effective sample sizes smaller than counts suggest. We measure and address this explicitly.

- **Weights.** Queen contiguity within a country, row-standardised: $W_{ij} = a_{ij} / \sum_k a_{ik}$.
- **Moran's I.** For a column $x$ with mean removed,

$$I = \frac{n}{\sum_{ij} W_{ij}} \cdot \frac{\sum_{ij} W_{ij}\, x_i x_j}{\sum_i x_i^2},$$

compared with its permutation distribution within countries (`morans_perm`).
- **Eigenvector spatial filtering (Griffith).** Candidate vectors are eigenvectors of the country-centred weight matrix with eigenvalue at least 25% of the largest. They are added greedily to a regression of the column until the remaining residual has standardised Moran's I below 1.645 (at most 20). The residual replaces the column, and the screen is re-run on filtered data.
- **Calibration.** Independent spatially smooth noise with Moran's I of about 0.16 and 0.57 is paired with real outcomes: the plain screen produces many false discoveries at strong autocorrelation (16.9 per simulation for levels at I about 0.57), the filtered screen essentially none.
- **Reading.** The filter also removes gradients that really are shared, such as a within-country North-South gradient. A pair that fails after filtering may be explained by that gradient or over-filtered, so both versions are reported.

## 9. High-frequency (monthly, quarterly) time series

The cross-section tools do not apply to time series. The rules here follow standard time-series practice (`analysis/hf.py`).

- **Stationarity first.** Each series is transformed (difference, log difference, or level if already stationary) and checked with the Augmented Dickey-Fuller test (null: unit root) and the KPSS test (null: stationary). We look for agreement of both; nothing is correlated in levels, since two trending series correlate spuriously.
- **Prewhitened cross-correlation.** If $x_t$ follows an AR($p$) process $\phi(L)x_t=u_t$ (order by BIC), we apply the same filter $\phi(L)$ to both series and correlate the filtered series (Box-Jenkins). This removes the autocorrelation that would otherwise produce spurious lagged correlations; the lag with the largest filtered correlation indicates lead or lag.
- **Granger causality** (a statement about predictability, not about causes). Does adding lags of $x$ to an autoregression of $y$ reduce the error? The panel version adds the dependent series' own lags, country and month effects and country-clustered errors.
- **Spread model.** $\Delta s_t = \alpha + \sum_j \beta_j z_{j,t-1} + \varepsilon_t$ for the change in the Italy-Germany yield spread on lagged, standardised differentials with Germany, estimated by OLS with **Newey-West (HAC)** standard errors (6 lags) separately by regime (2005-11, 2012-19, 2020 onwards), since the spread's behaviour changed with the sovereign debt crisis and with the ECB.
- **Event study.** For an event at month $t$, the statistic is the mean of the six months after minus the mean of the six months before, $S_t = \bar x_{[t+1,t+6]} - \bar x_{[t-6,t-1]}$. Its null distribution is $S$ computed at every other month more than six months from the event, giving an exact two-sided placebo p-value with +1 smoothing and a z-score $(S_t-\bar S)/\mathrm{sd}(S)$. BH runs over events times series.
- **Pass-through.** Wage growth (year on year) on the average of the current and previous four quarters of inflation plus unemployment, with country and quarter effects; the inflation coefficient is the cumulative pass-through in points of wage growth per point of inflation. Estimating the same regression country by country (Newey-West SEs) lets Italy be ranked.
- **Gaps.** Panels keep the full date grid so a row-based lag never silently jumps over a missing month.

## 10. Daily data: spreads and rating changes

`analysis/daily.py`. For an event on trading day $t$ (first trading day on or after the announcement) and horizon $h \in \{1,5,10\}$ days, the statistic is the change in the ITA-DEU spread over $[t-1, t+h]$, divided by a volatility scale

$$z = \frac{s_{t+h} - s_{t-1}}{\hat\sigma_{60}\,\sqrt{h+1}}$$

where $\hat\sigma_{60}$ is the trailing 60-day standard deviation of daily spread changes. The normalisation prevents 2011-12 and 2020 from dominating; a 20-day scale is the robustness check. Inference is by an **exact placebo distribution**: the same $z$ over all days whose window overlaps no event (event windows exclude $|i-p| \le h+1$, not just $h$). BH runs over every (event date, horizon). For rating changes there is also a pooled, direction-aligned test (mean of direction times $z$ against random non-event days with the same signs, one-sided) with robustness to the volatility scale, evidence level and the crisis years. The null runs the whole procedure on random dates and must find about nothing.

## 11. Latent structure, shift-share and distributional decompositions

- **PCA (L1, `latent.py`).** Principal components of the standardised country-by-indicator table (2015-19, 85 indicators after coverage and duplicate filters) by SVD. The number of components is chosen by **Horn's parallel analysis**: retain components whose eigenvalue exceeds the matching percentile of eigenvalues from column-shuffled data. Stability is tested by bootstrapping countries and by split-half loading congruence (2005-14 against 2015-24). Italy's score is read on fixed axes across windows and ranked among countries.
- **Industry panel (I1).** For each country, the OLS slope of industry productivity growth on an industry's ICT or intangible-capital intensity (the median of the other countries in the same KLEMS section, in an early period, so the regressor is not the country's own outcome). p-values come from permuting intensity across sections, because a clustered t-test over-rejects with about 14 clusters. Italy is ranked among the country slopes.
- **Shift-share (S1).** The gap between Italy's self-employment share and the pooled others' is split exactly into a *composition* term (sector mix) and a *within-sector* term: with sector shares $w_s$ and rates $r_s$,

$$\sum_s w^{IT}_s r^{IT}_s - \sum_s w^{O}_s r^{O}_s = \sum_s (w^{IT}_s - w^{O}_s)\, \bar r_s + \sum_s \bar w_s\,(r^{IT}_s - r^{O}_s),$$

with $\bar r_s$ and $\bar w_s$ the midpoints of the two groups. The decomposition is an identity, so it needs no test; Italy's two terms are ranked among countries.
- **Distribution (D1).** S80/S20, quintile shares, Gini before and after transfers; redistribution is the percentage fall of the Gini. Everything is ranked against every country's own gap to the median of the others.

## 12. Fiscal analyses

- **Budget-law coverage (`budget.py`).** Per law and horizon, with uses $U$ (impieghi) and resources $R$ (risorse) in the net-borrowing aggregate: balance $= R-U$, **coverage ratio** $=R/U$, deficit-financed share $=1-R/U$. Accounting identities (uses = higher spending + lower revenue, resources = higher revenue + lower spending) are checked on every parsed table (165+), so a parsing error shows up as a broken identity. These are ex-ante estimates from the government's technical reports, not realised effects.
- **Link to realised deficit and debt.** A pre-specified family of two predictors against three same-year outcomes, Spearman with permutation p (20,000 permutations), BH over the six, leave-one-year-out, without 2020-21, a lead placebo (the previous year's outcome should show nothing) and a year-shuffle null. With about 11 years, only large associations are detectable.
- **Deficit surprise ranked among the EU27 (`budget_xc.py`).** Surprise $=$ Eurostat outturn $-$ Commission autumn forecast (positive = worse); the idiosyncratic surprise subtracts the median of the other 26 members in the same year, removing common shocks. Italy's rank among 27 gives $p=\text{rank}/27$ (section 6). A decomposition into revenue, primary expenditure and interest is shown, and the GDP-denominator effect is separated as a level miss plus a denominator term. These decompositions are accounting identities and carry ranks only, with no new tests.
- **Contract timing (`contracts.py`).** Real change of the contractual wage index against the staleness of the agreement, with a branch-level permutation, **employee weights** (weighted rank correlation, Kish effective sample size $n_{\text{eff}} = (\sum w)^2/\sum w^2$ reported next to the p-value) and a calm-period placebo. The unit was changed after the first results (leaves to parent groups, because sub-sectors share agreements), and this is declared as a post-hoc choice in the framework document; later variants are labelled as variants of one question.

## 13. What the p-values and q-values mean here

- A **p-value** is the probability, under the null built by the stated shuffle, of a statistic at least as extreme. It is not the probability the hypothesis is true.
- A **q-value** is the smallest FDR at which the test would be a discovery. Within a family of $m$ tests, we expect about $q\times$ (number of discoveries) of them to be false.
- A **rank-based p** from a placebo ranking is bounded below by $1/N$ and is a descriptive ordering when $N$ is small.
- **Power.** With $n$ units only correlations of a certain size can be detected: for $n=27$ roughly $|\rho|\gtrsim 0.5$, for $n=21$-$23$ about 0.4. Failing to find a relation is never evidence that none exists; we state the detectable size next to null results.
- **Replication** across windows or vintages is the strongest evidence available, stronger than any single p-value; a pair that survives BH but not the split-sample test is reported as fragile.

## 14. Limitations that apply to everything

- Observational, small-$N$ data. No identification strategy; omitted variables and reverse causality are not excluded.
- Cross-country comparisons depend on harmonised definitions that change; breaks are flagged, not repaired.
- Italy is one unit: all Italy-specific statements are rankings.
- A large screen will surface pairs whose link is mechanical, definitional or driven by one episode; filters are heuristic and human reading remains part of the method.
- Flexibility in analysis (windows, transformations, filters) is a researcher degree of freedom. We limit it by fixing families beforehand, reporting all variants run, and requiring a calibrated null; we cannot eliminate it.
- Data revisions: the latest vintage is used, so outcomes such as Italy's deficit include later revisions (reported where material).

## 15. Reproducibility and checks

- `uv run python -m cifresignificative.fetch` and `clean.build` rebuild every table; `experiments.run_*_all()` rebuild every result under `data/processed/correlations/`; random procedures take a fixed seed (`np.random.default_rng(seed)`).
- Notebook cells are generated from `scripts/notebook_cells.py` and executed, so the text in a notebook matches its outputs.
- `tests/` covers the statistics directly: BH monotonicity and bounds, planted-correlation recovery, shuffled-null discovery counts, permutation calibration, spatial filter calibration, accounting identities of parsed tables. CI runs ruff, ty, the catalog freshness check and pytest.
- Hand-curated inputs in `data/external/` carry `evidence` labels (`verified`, `partly verified`, `unverified`); a date is `partly verified` only when two independent sources agree on the same day.

## Reading list by topic

| topic | reference |
|---|---|
| False discovery rate | Benjamini and Hochberg (1995), *J. Royal Statistical Society B* 57 |
| Rank correlation, Fisher transform | Spearman (1904); Fisher (1915) |
| Randomisation tests | Phipson and Smyth (2010), permutation p-values should never be zero |
| Clustered errors, panel models | Cameron and Miller (2015), *A Practitioner's Guide to Cluster-Robust Inference* |
| Prewhitening, ARMA | Box and Jenkins (1970) |
| HAC errors | Newey and West (1987) |
| Unit root and stationarity tests | Dickey and Fuller (1979); Kwiatkowski, Phillips, Schmidt and Shin (1992) |
| Granger causality | Granger (1969) |
| Parallel analysis | Horn (1965) |
| Spatial autocorrelation, eigenvector filtering | Moran (1950); Griffith (2003), *Spatial Autocorrelation and Spatial Filtering* |
| Beta-convergence | Barro and Sala-i-Martin (1992) |
| Decomposition | Oaxaca (1973), Blinder (1973) and shift-share literature |
