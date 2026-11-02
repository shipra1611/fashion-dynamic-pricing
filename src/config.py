from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; OUT = ROOT / "outputs"; BI = ROOT / "powerbi"
SEED = 42
# H&M data: 2019-09-22 .. 2020-09-22. First and last ISO weeks hold 1-2 days only -> dropped.
FIRST_FULL_WEEK = "2019-09-23"; LAST_FULL_WEEK = "2020-09-14"
N_CATEGORIES = 15           # top 14 product types by units + "Other"
# modelling-set filters (series = article x channel)
MIN_WEEKS, MIN_UNITS, MIN_PRICE_RANGE = 12, 30, 0.05
# demand model split: test = last 12 full weeks
TEST_START = "2020-06-29"
# --- Simulated A/B test: 6 full weeks test, 6 weeks pre-period ---
PRE_START, PRE_END = "2020-06-29", "2020-08-03"      # 6 weeks
AB_START, AB_END = "2020-08-10", "2020-09-14"        # 6 weeks
N_AB = 2400
# --- ASSUMPTIONS (no cost / inventory data in the dataset) ---
COST_RATIO = 0.50           # unit cost as share of list price (90th pct price of the series)
PRICE_BOUND = 0.15          # recommended price within +-15% of base price
TRUTH_ARTICLE_SD = 0.5      # simulated article-level elasticity heterogeneity
TRUTH_CATEGORY_SD = 0.3     # simulated category-level misspecification of the fitted elasticity
