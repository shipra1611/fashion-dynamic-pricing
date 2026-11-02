"""Power BI tables and charts."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *
p = pd.read_parquet(DATA / "panel.parquet", columns=["article_id","channel","week","units","price","category"]); p["revenue"] = p.units * p.price
w = p.groupby(["category","week"]).agg(units=("units","sum"), revenue=("revenue","sum"), active_articles=("article_id","nunique")).reset_index()
w["avg_price"] = w.revenue / w.units; w["price_index_vs_category_mean"] = w.avg_price / w.groupby("category").avg_price.transform("mean")
w.assign(**{c: w[c].round(5) for c in ["avg_price","price_index_vs_category_mean","revenue"]}).to_csv(BI / "weekly_category_trends.csv", index=False)
print("weekly_category_trends:", w.shape, "| categories", w.category.nunique())

el = pd.read_csv(BI / "category_elasticity.csv"); on = el[el.channel == "online"].sort_values("elasticity")
est = pd.read_csv(OUT / "ab_rerandomisation_estimates.csv"); sm = pd.read_csv(BI / "ab_rerandomisation_summary.csv").set_index("metric")
dm = pd.read_csv(BI / "demand_model_metrics.csv")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
y = np.arange(len(on)); ax[0].errorbar(on.elasticity, y, xerr=[on.elasticity - on.ci_lo, on.ci_hi - on.elasticity], fmt="o", capsize=3)
ax[0].axvline(-1, ls="--", c="grey"); ax[0].axvline(0, c="k", lw=.5); ax[0].set_yticks(y); ax[0].set_yticklabels(on.category, fontsize=8)
ax[0].set_title("Online price elasticity by category (95% CI)\nbelow -1 = elastic"); ax[0].set_xlabel("elasticity")
e = est[est.metric == "rev"]; ax[1].hist(e.c_lift, bins=40, alpha=.8); ax[1].axvline(sm.loc["rev", "true_effect_pct"], c="r", label="true effect (simulated world)")
ax[1].axvline(0, c="k", lw=.6); ax[1].set_title("SIMULATED A/B: revenue lift estimate\nover 1,000 re-randomisations (CUPED)"); ax[1].set_xlabel("estimated lift %"); ax[1].legend(fontsize=8)
lab = ["Naive\n(last week)", "XGB A\nprice+article", "XGB B\n+ past sales"]; ax[2].bar(lab, dm.R2_log_units, label="R2 log units"); ax[2].bar(lab, dm.R2_units, width=.4, label="R2 units")
for i, v in enumerate(dm.R2_log_units): ax[2].text(i, v + .01, f"{v:.2f}", ha="center", fontsize=8)
ax[2].set_title("Demand model, test = last 12 weeks"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "pricing_overview.png", dpi=150)
