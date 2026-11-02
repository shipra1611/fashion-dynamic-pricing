"""Weekly demand model for article x channel x week (weeks with at least one sale). Train: before 2020-06-29. Test: the last 12 full weeks.
A = price + static article features + age (no past sales). B = A + past sales (lag-1, 4-week rolling, series mean in training).
Reported honestly: B's high R2 comes mostly from past sales, not price."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
from config import *

d = pd.read_parquet(DATA / "panel.parquet"); d = d[d.eligible].sort_values(["article_id","channel","week"]).copy()
d["wk"] = ((d.week - pd.Timestamp(FIRST_FULL_WEEK)).dt.days // 7).astype(int)
g = d.groupby(["article_id","channel"])
d["lag1_log"] = g.log_units.shift(1); d["gap_wk"] = d.wk - g.wk.shift(1)
d["roll4_log"] = g.log_units.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
d["lag1_price_rel"] = g.log_price_rel.shift(1); d["price_change"] = d.log_price_rel - d.lag1_price_rel
d["cat_code"] = pd.factorize(d.category)[0]
tr = d.week < TEST_START; te = ~tr
d["sid"] = g.ngroup()
sm = d[tr].groupby("sid").log_units.mean(); d["series_mean_train"] = d.sid.map(sm)
# price references from TRAINING weeks only (the stored full-period median / 90th pct would leak future prices)
d["series_median_price"] = d.sid.map(d[tr].groupby("sid").price.median()); d["list_price"] = d.sid.map(d[tr].groupby("sid").price.quantile(.9))
d["log_price_rel"] = np.log(d.price / d.series_median_price); d["discount_depth"] = (1 - d.price / d.list_price).clip(lower=0)
d["lag1_price_rel"] = d.groupby("sid").log_price_rel.shift(1); d["price_change"] = d.log_price_rel - d.lag1_price_rel

A = ["log_price_rel","discount_depth","price","list_price","channel","cat_code","department_no","colour_group_code","garment_group_no","age_weeks"]
B = A + ["lag1_log","gap_wk","roll4_log","series_mean_train","price_change"]
P = dict(n_estimators=400, learning_rate=0.06, max_depth=7, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=5, tree_method="hist", random_state=SEED, n_jobs=-1)

def evaluate(feats, label, need=None):
    m = d[tr & d[need].notna().all(axis=1)] if need else d[tr]; t = d[te & d[need].notna().all(axis=1)] if need else d[te]
    mdl = xgb.XGBRegressor(**P).fit(m[feats], m.log_units)
    pl = mdl.predict(t[feats])
    pu = np.exp(pl) * np.mean(np.exp(m.log_units - mdl.predict(m[feats])))        # smearing correction for back-transform
    return mdl, t, {"model": label, "n_train": len(m), "n_test": len(t), "R2_log_units": r2_score(t.log_units, pl),
                    "R2_units": r2_score(t.units, pu), "MAE_units": mean_absolute_error(t.units, pu)}
rows = []
need = ["lag1_log", "series_mean_train"]
tA = d[te & d[need].notna().all(axis=1)]
# naive: last observed week's units
nv = tA.assign(p=np.exp(tA.lag1_log)); rows.append({"model": "Naive (last observed week)", "n_train": 0, "n_test": len(nv), "R2_log_units": r2_score(nv.log_units, nv.lag1_log),
        "R2_units": r2_score(nv.units, nv.p), "MAE_units": mean_absolute_error(nv.units, nv.p)})
mA, tA2, rA = evaluate(A, "XGBoost A: price + article + age (no past sales)", need); rows.append(rA)
mB, tB2, rB = evaluate(B, "XGBoost B: A + past sales", need); rows.append(rB)
res = pd.DataFrame(rows).round(4); print(res.to_string(index=False)); res.to_csv(BI / "demand_model_metrics.csv", index=False)

# how much does price matter? shuffle price features within test (model A)
rng = np.random.default_rng(SEED); base = r2_score(tA2.log_units, mA.predict(tA2[A])); out = {}
for f in ["log_price_rel", "discount_depth", "price"]:
    t = tA2.copy(); t[f] = rng.permutation(t[f].to_numpy()); out[f] = base - r2_score(t.log_units, mA.predict(t[A]))
t = tA2.copy()
for f in ["log_price_rel","discount_depth","price"]: t[f] = rng.permutation(t[f].to_numpy())
print("model A R2(log units) drop when each price feature is shuffled:", {k: round(v, 4) for k, v in out.items()}, "| all three together:", round(base - r2_score(t.log_units, mA.predict(t[A])), 4))
pd.Series({**out, "all_price": base - r2_score(t.log_units, mA.predict(t[A]))}).round(4).to_csv(OUT / "demand_price_importance.csv", header=["r2_drop"])
