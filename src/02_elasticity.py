"""Price elasticity of demand with a conditional-Poisson (PPML fixed-effects) model.

  E[units_it] = exp(beta * log(price_it / series median) + series FE_i + week FE_t + age-bin FE_a(it))

Series = article x channel. Zero-sales weeks inside a series' active window (first to last observed sale) are filled with
units = 0 and the last observed price carried forward (ASSUMPTION: the shelf price persists in a week with no sale).
Age bins absorb the lifecycle (prices fall as articles age while demand also decays), which otherwise biases beta toward zero.
Standard errors are clustered by article. beta is an elasticity (units respond to price by beta percent per 1 percent).

Main specification was fixed BEFORE looking at results: Poisson with zeros, week FE and age bins.
Still observational: markdown timing is not randomised, so this is an association, not a causal effect.
"""
import numpy as np, pandas as pd, scipy.sparse as sp
from config import *

d = pd.read_parquet(DATA / "panel.parquet"); d = d[d.eligible].copy()
d["wk"] = ((d.week - pd.Timestamp(FIRST_FULL_WEEK)).dt.days // 7).astype(int)
d["sid"] = pd.factorize(d.article_id.astype(str) + "_" + d.channel.astype(str))[0]

# ---- fill zero-sales weeks inside each series' window ----
lo = d.groupby("sid").wk.transform("min"); hi = d.groupby("sid").wk.transform("max")
first = d.drop_duplicates("sid").assign(lo=lambda x: x.wk.groupby(x.sid).transform("min"))
keep = ["sid","article_id","channel","category","series_median_price","list_price"]
meta = d.drop_duplicates("sid")[keep].set_index("sid")
rng_ = d.groupby("sid").wk.agg(["min", "max"])
grid = pd.DataFrame([(s, w) for s, (a, b) in zip(rng_.index, rng_.to_numpy()) for w in range(a, b + 1)], columns=["sid","wk"])
f = grid.merge(d[["sid","wk","units","price"]], on=["sid","wk"], how="left")
f["observed"] = f.units.notna(); f["units"] = f.units.fillna(0.0)
f["price"] = f.groupby("sid").price.ffill()
f = f.join(meta, on="sid")
f["log_price_rel"] = np.log(f.price / f.series_median_price)
f["age_wk"] = f.wk - f.groupby("article_id").wk.transform("min")
f["age_bin"] = pd.cut(f.age_wk, [-1, 0, 1, 2, 3, 5, 8, 12, 20, 30, 60], labels=False)
print(f"observed rows {int(f.observed.sum()):,} | after zero-fill {len(f):,} (zero weeks {(~f.observed).sum():,}, {(~f.observed).mean():.1%}) | series {f.sid.nunique():,}")
f.to_parquet(DATA / "elasticity_grid.parquet", index=False)

def design(df, week=True, age=True, extra=()):
    n = len(df); cols = [sp.csr_matrix(df[["log_price_rel", *extra]].to_numpy(float))]; names = ["log_price", *extra]
    if week:
        w = pd.factorize(df.wk)[0]; k = w.max() + 1; m = sp.csr_matrix((np.ones(n), (np.arange(n), w)), shape=(n, k))[:, 1:]; cols.append(m); names += [f"w{i}" for i in range(1, k)]
    if age:
        a = df.age_bin.to_numpy(int); k = a.max() + 1; m = sp.csr_matrix((np.ones(n), (np.arange(n), a)), shape=(n, k))[:, 1:]; cols.append(m); names += [f"a{i}" for i in range(1, k)]
    return sp.hstack(cols).tocsr(), names

def cond_poisson(df, **kw):
    Z, names = design(df, **kw); y = df.units.to_numpy(float)
    s = pd.factorize(df.sid)[0]; S = sp.csr_matrix((np.ones(len(df)), (s, np.arange(len(df)))))   # series x rows
    cl = pd.factorize(df.article_id)[0]; C = sp.csr_matrix((np.ones(len(df)), (cl, np.arange(len(df)))))
    ni = np.asarray(S @ y).ravel(); th = np.zeros(Z.shape[1])
    for it in range(30):
        eta = Z @ th; e = np.exp(eta - eta.max()); E = np.asarray(S @ e).ravel(); pi = e / E[s]; mu = ni[s] * pi
        g = Z.T @ (y - mu)
        M = np.asarray((S @ Z.multiply(pi[:, None]).tocsr()).todense())                      # series x k
        H = (Z.T @ Z.multiply(mu[:, None]).tocsr()).toarray() - (M * ni[:, None]).T @ M
        step = np.linalg.solve(H + 1e-8 * np.eye(len(th)), g); th = th + step
        if np.abs(step).max() < 1e-7: break
    R = Z.multiply((y - mu)[:, None]).tocsr(); Sc = np.asarray((C @ R).todense()); G = Sc.shape[0]
    Hi = np.linalg.inv(H + 1e-8 * np.eye(len(th))); V = Hi @ (Sc.T @ Sc) @ Hi * G / (G - 1)
    return th[0], np.sqrt(V[0, 0]), len(df), G

rows = []
def add(label, df, **kw):
    b, se, n, G = cond_poisson(df, **kw)
    rows.append({"spec": label, "elasticity": b, "se": se, "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se, "rows": n, "articles": G}); print(rows[-1]["spec"], round(b, 3), round(se, 3), flush=True)
add("A. Poisson FE, zeros filled, week FE only", f, age=False)
add("B. + age bins (MAIN)", f)
for ch, g in f.groupby("channel"): add(f"B by channel {ch} ({'online' if ch == 2 else 'store'})", g)
add("B on observed weeks only (no zero fill)", f[f.observed])
rob = pd.DataFrame(rows).round(4); rob.to_csv(OUT / "elasticity_robustness.csv", index=False)

# Category elasticities. Store-channel estimates are positive (implausible for apparel, residual confounding), so the
# elasticities used for pricing come from the ONLINE channel. This choice was made after seeing the channel results.
def by_cat(df, tag):
    out = []
    for c, g in df.groupby("category"):
        b, se, n, G = cond_poisson(g)
        out.append({"channel": tag, "category": c, "elasticity": b, "se": se, "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se, "rows": n, "articles": G, "units": int(g.units.sum())})
    return pd.DataFrame(out)
on = by_cat(f[f.channel == 2], "online"); st = by_cat(f[f.channel == 1], "store")
cat = pd.concat([on, st]).sort_values(["channel", "elasticity"]); cat["used_for_pricing"] = cat.channel == "online"
print("\nBy category, ONLINE (used for pricing):\n", on.sort_values("elasticity").round(3).to_string(index=False))
print("\nBy category, STORE (not used):\n", st.sort_values("elasticity").round(3)[["category","elasticity","se"]].to_string(index=False))
cat.round(4).to_csv(BI / "category_elasticity.csv", index=False)
