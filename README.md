# Dynamic Pricing Optimization for Fashion Retail (H&M)

Python, Poisson fixed-effects regression, XGBoost, A/B test design (CUPED, power analysis), Power BI-ready tables.
Data: H&M Personalized Fashion Recommendations (Kaggle), 2019-09-23 to 2020-09-14, 52 full weeks. The `price` field in this dataset is anonymised (scaled), so only relative prices are meaningful.

## What is real and what is simulated
| Part | Status |
|---|---|
| Panel, price variation, elasticities, demand model | **Real data** |
| Unit costs, inventory | **Not in the dataset.** Unit cost = 50% of list price is an assumption (`config.py`). There is no inventory data; COGS of units sold is used as a stand-in for replenishment spend |
| A/B test: article pool, assignment, pre-period, control outcomes, variance, power analysis | **Real data** |
| A/B test: the treated arm's response | **Simulated** from the fitted elasticities plus assumed noise (`config.py`). The measured "lift" is a property of those assumptions, not a finding about H&M |

## Setup
Kaggle's H&M files are too large to use directly. `run_all.sh` expects two small inputs in `data/`:
- `articles.csv` (copy from the competition)
- `hm_weekly_channel.csv.gz`: transactions from 2019-09-22 aggregated to article x week x sales channel (units, revenue, price), produced with a short chunked script in a Kaggle notebook.
Then: `pip install -r requirements.txt && ./run_all.sh`

## Pipeline
| Step | File | Output |
|---|---|---|
| Panel + 15 categories | `src/01_prepare.py` | `data/panel.parquet` |
| Price elasticity | `src/02_elasticity.py` | `powerbi/category_elasticity.csv`, `outputs/elasticity_robustness.csv` |
| Demand model | `src/03_demand_model.py` | `powerbi/demand_model_metrics.csv` |
| Repricing policy + simulated A/B | `src/04_pricing_ab.py` | `powerbi/price_recommendations.csv`, `ab_*.csv` |
| Power BI tables, charts | `src/05_exports_charts.py` | `powerbi/weekly_category_trends.csv`, `outputs/pricing_overview.png` |

Categories: the 14 most-sold product types plus "Other" = 15 (defined by volume, not by H&M's own hierarchy).
Cleaning: the first and last weeks of the extract hold only 1-2 days of sales and were dropped (they would look like demand collapses).

## Elasticity
Model: `units ~ exp(beta * log price relative to the article's median + series FE + week FE + age-bin FE)`, fitted by conditional Poisson (PPML). A series is an article x channel. Zero-sales weeks inside an article's active window are filled with 0 units and the last price carried forward (an assumption). Errors clustered by article. 27,703 articles, 36,211 series, 1.24M rows (26.5% zero weeks).

| Specification | Elasticity |
|---|---|
| Naive log-OLS on sold weeks, pooled | -0.04 |
| Poisson, zeros filled, week FE | -0.13 |
| **+ age controls (main, fixed before looking at results), pooled** | **-0.25** (SE 0.02) |
| Same, online channel | **-0.60** |
| Same, store channel | **+0.30** (implausible, residual confounding) |

The naive estimate is badly biased toward zero: weeks with no sale are missing and prices fall as articles age while demand also decays. Category elasticities use the **online channel only**; that choice was made after seeing the store estimates were positive, so treat it as a judgment call. Store estimates are saved but not used.

Online elasticity by category (95% CI): Sweater -1.82 (-2.06, -1.58), Dress -1.26 (-1.40, -1.12), Skirt -1.21 (-1.45, -0.98), Shorts -0.92, Bikini top -0.85, Blouse -0.82, Top -0.79, Other -0.76, Trousers -0.75, Leggings -0.64, Bra -0.50, Swimwear bottom -0.37, Vest top -0.17, T-shirt -0.06, Underwear bottom +0.35 (-0.02, 0.73). Knitwear, dresses and skirts are price-sensitive; basics are not.
These are associations: markdowns are not randomised, so they are not causal estimates.

## Demand model
Predicts weekly units for article-weeks with at least one sale. Train before 2020-06-29, test = the last 12 weeks (203,600 rows). Price reference features use training weeks only (a first version leaked future prices and was fixed).

| Model | R2 (log units) | R2 (units) |
|---|---|---|
| Naive: last observed week | 0.62 | 0.46 |
| XGBoost A: price + article + age, no past sales | 0.43 | 0.17 |
| **XGBoost B: A + past sales** | **0.73** | **0.53** |

Most of B's strength comes from past sales; price alone (A) is clearly informative (shuffling the price features drops A's R2 by 0.51). The test window includes COVID-era demand shifts.

## Repricing policy
Revenue objective, +-15% around base price, never below unit cost. If a category's online elasticity CI is entirely below -1 (elastic) the price goes down; entirely above -1 (inelastic) it goes up; if the CI contains -1 it is left unchanged. For 11,679 active articles: 6,943 raise, 2,676 hold, 2,060 lower (Dress and Sweater). The fitted elasticities imply about +4.8% revenue.

## Simulated A/B test (2,400 articles, 6 weeks)
2,400 articles sampled and randomised 50/50 within category on pre-period information only (pre-period balance: SMD -0.01). Outcomes: 6-week revenue, gross profit, units, COGS. Analysis: difference in means and CUPED using pre-period outcomes.

**The true effect in the simulated world is +5.9% revenue and +16.5% gross profit.** The estimator is fine (mean over 1,000 re-randomisations 6.5% vs true 5.9%, 94% CI coverage), but the design is **underpowered**:
- Only **11.7%** of re-randomisations detect the revenue effect (40.6% for gross profit). The single draw reported in `ab_test_results.csv` (+17.9% revenue, p=0.02) is a lucky overestimate, about 3x the truth.
- Revenue per article is extremely skewed (coefficient of variation 2.3). With 1,200 articles per arm the minimum detectable revenue lift is 26.7% (18.6% with CUPED). Detecting a 5% lift needs about 16,600 articles per arm with CUPED (34,200 without).
- If true elasticities are 2x the fitted values, the true revenue effect shrinks to +1.6% (gross profit +10%); see `ab_sensitivity.csv`.
- Gross profit depends heavily on the assumed 50% cost ratio.

