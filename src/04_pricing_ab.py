"""Price recommendations + a SIMULATED 6-week A/B test on 2,400 articles.

What is real: the article pool, pre-period and test-window demand (observed H&M weeks), randomisation, the analysis pipeline,
and the power analysis (variance comes from real data).
What is simulated: the treatment arm's response. Treated demand = observed demand x r^e_true, where e_true is the fitted category
elasticity PLUS assumed heterogeneity/misspecification (config). The measured lift is therefore a property of those assumptions.
Policy (pre-specified): revenue objective. If the category's online-elasticity 95% CI excludes -1: elastic (<-1) -> lower price to the
floor, inelastic (>-1) -> raise to the cap; if the CI covers -1 -> leave price unchanged. Bounds +-15% around base price; price never below unit cost.
Unit cost = COST_RATIO x article list price (ASSUMPTION; dataset has no costs)."""
import numpy as np, pandas as pd
from scipy import stats
from config import *
rng = np.random.default_rng(SEED)

p = pd.read_parquet(DATA / "panel.parquet", columns=["article_id","channel","week","units","price","list_price","category"])
p["revenue"] = p.units * p.price
el = pd.read_csv(BI / "category_elasticity.csv"); el = el[el.channel == "online"].set_index("category")
weeks = lambda a, b: pd.date_range(a, b, freq="W-MON")
PRE, AB = weeks(PRE_START, PRE_END), weeks(AB_START, AB_END); assert len(PRE) == 6 and len(AB) == 6

aw = p.groupby(["article_id","week"]).agg(units=("units","sum"), revenue=("revenue","sum")).reset_index()
def window(ws, tag):
    x = aw[aw.week.isin(ws)].groupby("article_id").agg(**{f"units_{tag}": ("units","sum"), f"revenue_{tag}": ("revenue","sum"), f"wk_{tag}": ("week","nunique")}); return x
art = p.groupby("article_id").agg(category=("category","first")).join(window(PRE, "pre")).join(window(AB, "ab"))
lp = p.assign(w=p.units * p.list_price).groupby("article_id").agg(w=("w","sum"), u=("units","sum")); art["list_price"] = lp.w / lp.u
art = art.fillna({"units_ab": 0, "revenue_ab": 0, "wk_ab": 0})
pool = art[(art.wk_pre >= 4) & (art.units_pre >= 20)].copy()          # eligibility uses PRE-period only (no peeking at the test window)
pool["base_price"] = pool.revenue_pre / pool.units_pre; pool["unit_cost"] = COST_RATIO * pool.list_price
print(f"articles active in pre-period (>=4 of 6 weeks, >=20 units): {len(pool):,}")

# ---- policy ----
e = pool.category.map(el.elasticity); lo_ci = pool.category.map(el.ci_lo); hi_ci = pool.category.map(el.ci_hi)
B = PRICE_BOUND
r_lo = np.maximum(1 - B, np.minimum(pool.unit_cost / pool.base_price, 1 + B)); r_hi = pd.Series(1 + B, index=pool.index)
pool["action"] = np.select([hi_ci < -1, lo_ci > -1], ["lower", "raise"], default="hold")
pool["price_ratio"] = np.select([pool.action == "lower", pool.action == "raise"], [r_lo, r_hi], default=1.0)
pool["rec_price"] = pool.base_price * pool.price_ratio; pool["elasticity_used"] = e
pool["expected_rev_change_pct"] = 100 * (pool.price_ratio ** (1 + e) - 1)
print("policy actions:", pool.action.value_counts().to_dict())
print(pool.groupby("category").agg(articles=("action","size"), action=("action","first"), elasticity=("elasticity_used","first")).round(2).to_string())
pool.round(4).reset_index().to_csv(BI / "price_recommendations.csv", index=False)

# ---- sample 2,400 stratified by category, randomise 50/50 within category ----
n = min(N_AB, len(pool)); share = pool.category.value_counts(normalize=True)
k = (share * n).round().astype(int); k.iloc[0] += n - k.sum()
smp = pd.concat([pool[pool.category == c].sample(min(int(m), (pool.category == c).sum()), random_state=SEED) for c, m in k.items()])
smp["arm"] = ""
for c, g in smp.groupby("category"):
    idx = rng.permutation(g.index.to_numpy()); half = len(idx) // 2; smp.loc[idx[:half], "arm"] = "treatment"; smp.loc[idx[half:], "arm"] = "control"
print(f"A/B sample {len(smp):,}: " + str(smp.arm.value_counts().to_dict()))

# ---- potential outcomes: Y0 = observed, Y1 = simulated outcome if the article were priced at the recommended ratio ----
wk_all = aw[aw.article_id.isin(smp.index) & aw.week.isin(AB)].set_index("article_id")
def potential_outcomes(scale=1.0, bias=0.0, seed=SEED):
    r = np.random.default_rng(seed); s_ = smp.copy()
    cat_shift = s_.category.map({c: r.normal(0, TRUTH_CATEGORY_SD) for c in s_.category.unique()})
    s_["true_e"] = np.minimum(s_.elasticity_used * scale + bias + cat_shift + r.normal(0, TRUTH_ARTICLE_SD, len(s_)), -0.02)
    wk = wk_all.copy(); wk["m"] = (s_.price_ratio ** s_.true_e).reindex(wk.index).to_numpy(); u = wk.units.astype(int)
    down = wk.m <= 1; wk["u1"] = u
    wk.loc[down, "u1"] = r.binomial(u[down], wk.loc[down, "m"].clip(0, 1))
    wk.loc[~down, "u1"] = u[~down] + r.poisson(u[~down] * (wk.loc[~down, "m"] - 1))
    ratio = s_.price_ratio.reindex(wk.index).to_numpy(); cost = s_.unit_cost.reindex(wk.index).to_numpy(); px = wk.revenue / wk.units
    wk["rev1"] = wk.u1 * px * ratio; wk["gp1"] = wk.rev1 - wk.u1 * cost; wk["cogs1"] = wk.u1 * cost
    wk["gp0"] = wk.revenue - wk.units * cost; wk["cogs0"] = wk.units * cost
    ag = wk.groupby(level=0).agg(units0=("units","sum"), rev0=("revenue","sum"), gp0=("gp0","sum"), cogs0=("cogs0","sum"), units1=("u1","sum"), rev1=("rev1","sum"), gp1=("gp1","sum"), cogs1=("cogs1","sum"))
    s_ = s_.join(ag).fillna(0.0)
    s_["gp_pre"] = s_.revenue_pre - s_.units_pre * s_.unit_cost; s_["cogs_pre"] = s_.units_pre * s_.unit_cost
    return s_
PRE_COL = {"rev": "revenue_pre", "gp": "gp_pre", "cogs": "cogs_pre", "units": "units_pre"}

def estimate(y, arm_t, pre):
    """arm_t boolean array. Returns raw and CUPED difference in means as % of control mean, with SE-based CI/p."""
    t, c = y[arm_t], y[~arm_t]; d = t.mean() - c.mean(); se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    th = np.cov(y, pre)[0, 1] / pre.var(); adj = y - th * (pre - pre.mean()); ta, ca = adj[arm_t], adj[~arm_t]
    dc = ta.mean() - ca.mean(); sec = np.sqrt(ta.var(ddof=1) / len(ta) + ca.var(ddof=1) / len(ca)); m = c.mean()
    return dict(lift=100 * d / m, lo=100 * (d - 1.96 * se) / m, hi=100 * (d + 1.96 * se) / m, p=2 * (1 - stats.norm.cdf(abs(d / se))),
                c_lift=100 * dc / m, c_lo=100 * (dc - 1.96 * sec) / m, c_hi=100 * (dc + 1.96 * sec) / m, c_p=2 * (1 - stats.norm.cdf(abs(dc / sec))), vr=100 * (1 - adj.var() / y.var()))

def rerandomise(arm_cats, n_rep, rr):
    """stratified 50/50 within category; yields boolean treatment vectors"""
    groups = [np.where(arm_cats == c)[0] for c in np.unique(arm_cats)]
    for _ in range(n_rep):
        t = np.zeros(len(arm_cats), bool)
        for g in groups: t[rr.permutation(g)[: len(g) // 2]] = True
        yield t

S = potential_outcomes()
cats = S.category.to_numpy(); main_arm = (S.arm == "treatment").to_numpy()
print(f"\nSIMULATED WORLD, 2,400 articles. TRUE sample effect of the policy (all articles treated vs untreated, in the simulation):")
true = {m: 100 * (S[f"{m}1"].sum() - S[f"{m}0"].sum()) / S[f"{m}0"].sum() for m in ["rev", "gp", "cogs", "units"]}
print({k: round(v, 2) for k, v in true.items()})
exp_rev = 100 * ((S.price_ratio ** (1 + S.elasticity_used)) * S.rev0).sum() / S.rev0.sum() - 100
print(f"optimizer's own expected revenue change from the fitted elasticities: {exp_rev:+.2f}%")

# one randomisation (the 'experiment you would actually run')
smd = lambda col: (S[main_arm][col].mean() - S[~main_arm][col].mean()) / np.sqrt((S[main_arm][col].var() + S[~main_arm][col].var()) / 2)
print(f"balance on PRE-period revenue (SMD): {smd('revenue_pre'):+.3f}")
one = []
for m in ["rev", "gp", "cogs", "units"]:
    y = np.where(main_arm, S[f"{m}1"], S[f"{m}0"]); r = estimate(y, main_arm, S[PRE_COL[m]].to_numpy()); r["metric"] = m; r["true_effect_pct"] = true[m]; one.append(r)
one = pd.DataFrame(one)[["metric", "true_effect_pct", "lift", "lo", "hi", "p", "c_lift", "c_lo", "c_hi", "c_p", "vr"]]; one.round(3).to_csv(BI / "ab_test_results.csv", index=False)
print("\nONE randomisation (this is what a single real test would show; compare with the true effect):\n", one.round(3).to_string(index=False))

# 1,000 re-randomisations: what does this test design deliver?
rr = np.random.default_rng(SEED + 1); rep = {m: [] for m in ["rev", "gp"]}
for t in rerandomise(cats, 1000, rr):
    for m in rep: rep[m].append(estimate(np.where(t, S[f"{m}1"], S[f"{m}0"]), t, S[PRE_COL[m]].to_numpy()))
pd.concat({m: pd.DataFrame(l) for m, l in rep.items()}, names=["metric","rep"]).reset_index().to_csv(OUT / "ab_rerandomisation_estimates.csv", index=False)
sumry = []
for m, lst in rep.items():
    df = pd.DataFrame(lst); T = true[m]
    sumry.append({"metric": m, "true_effect_pct": T, "mean_estimate_raw": df.lift.mean(), "sd_estimate_raw": df.lift.std(), "mean_estimate_cuped": df.c_lift.mean(), "sd_estimate_cuped": df.c_lift.std(),
                  "share_significant_raw_pct": 100 * (df.p < .05).mean(), "share_significant_cuped_pct": 100 * (df.c_p < .05).mean(),
                  "share_wrong_sign_and_significant_cuped_pct": 100 * ((df.c_p < .05) & (np.sign(df.c_lift) != np.sign(T))).mean(),
                  "ci_coverage_cuped_pct": 100 * ((df.c_lo <= T) & (T <= df.c_hi)).mean()})
sumry = pd.DataFrame(sumry).round(2); sumry.to_csv(BI / "ab_rerandomisation_summary.csv", index=False)
print("\n1,000 re-randomisations of the same 2,400 articles:\n", sumry.T.to_string(header=False))

# sensitivity to how wrong the fitted elasticity is (true = scale x fitted + bias)
sens = []
for sc, bi in [(0.5, 0), (1.0, 0), (1.5, 0), (2.0, 0), (1.0, -0.5), (1.0, 0.5)]:
    Sx = potential_outcomes(sc, bi, seed=SEED + 7); rr2 = np.random.default_rng(SEED + 2); res_ = []
    for t in rerandomise(cats, 300, rr2): res_.append(estimate(np.where(t, Sx.rev1, Sx.rev0), t, Sx.revenue_pre.to_numpy()))
    df = pd.DataFrame(res_); T = 100 * (Sx.rev1.sum() - Sx.rev0.sum()) / Sx.rev0.sum(); G = 100 * (Sx.gp1.sum() - Sx.gp0.sum()) / Sx.gp0.sum()
    sens.append({"true_elasticity": f"{sc}x fitted" + (f" {bi:+.1f}" if bi else ""), "true_revenue_effect_pct": T, "true_gross_profit_effect_pct": G, "mean_cuped_estimate_pct": df.c_lift.mean(), "share_significant_cuped_pct": 100 * (df.c_p < .05).mean()})
sens = pd.DataFrame(sens).round(2); sens.to_csv(BI / "ab_sensitivity.csv", index=False); print("\nSensitivity to the TRUE elasticity (300 re-randomisations each):\n", sens.to_string(index=False))
S.round(4).reset_index().to_csv(BI / "ab_test_articles.csv", index=False)

# ---- power analysis from REAL variance ----
base = S[~main_arm]; rows = []
for y, pre in [("rev0", "revenue_pre"), ("gp0", "gp_pre")]:
    mu, sd = base[y].mean(), base[y].std(); rho = base[[y, pre]].corr().iloc[0, 1]
    for npa in [300, 600, 1200, 2400, 5000, 10000]:
        for cuped in (False, True):
            sdv = sd * (np.sqrt(1 - rho ** 2) if cuped else 1)
            rows.append({"metric": y, "n_per_arm": npa, "cuped": cuped, "pre_post_corr": rho, "min_detectable_lift_pct": 100 * (stats.norm.ppf(.975) + stats.norm.ppf(.8)) * sdv * np.sqrt(2 / npa) / mu})
pw = pd.DataFrame(rows).round(3); pw.to_csv(BI / "ab_power_analysis.csv", index=False)
mu, sd = base.rev0.mean(), base.rev0.std(); rho = base[["rev0", "revenue_pre"]].corr().iloc[0, 1]
need = pd.DataFrame([{"target_lift_pct": L, "n_per_arm_no_cuped": int(2 * ((stats.norm.ppf(.975) + stats.norm.ppf(.8)) * sd / (L / 100 * mu)) ** 2), "n_per_arm_cuped": int(2 * ((stats.norm.ppf(.975) + stats.norm.ppf(.8)) * sd * np.sqrt(1 - rho ** 2) / (L / 100 * mu)) ** 2)} for L in (3, 5, 10, 15)])
need.to_csv(BI / "ab_required_sample_size.csv", index=False)
print("\nMinimum detectable REVENUE lift (80% power), real variance:\n", pw[pw.metric == "rev0"].pivot(index="n_per_arm", columns="cuped", values="min_detectable_lift_pct").to_string())
print("\nArticles per arm needed to detect a given revenue lift:\n", need.to_string(index=False)); print("pre/post corr %.3f | CV of 6-week revenue per article %.2f" % (rho, sd / mu))
