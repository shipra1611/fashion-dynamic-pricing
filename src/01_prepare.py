"""Build the article x channel x week panel with categories and price features."""
import numpy as np, pandas as pd
from config import *
a = pd.read_csv(DATA / "articles.csv", usecols=["article_id","product_type_name","product_group_name","department_no","index_group_name","colour_group_code","garment_group_no"])
d = pd.read_csv(DATA / "hm_weekly_channel.csv.gz", parse_dates=["week"])
n0 = len(d)
d = d[(d.week >= FIRST_FULL_WEEK) & (d.week <= LAST_FULL_WEEK)].copy()
print(f"rows {n0:,} -> {len(d):,} after dropping the two partial weeks | weeks {d.week.nunique()} | articles {d.article_id.nunique():,}")

top = (d.merge(a[["article_id","product_type_name"]], on="article_id").groupby("product_type_name").units.sum()
         .sort_values(ascending=False).head(N_CATEGORIES - 1).index)
a["category"] = np.where(a.product_type_name.isin(top), a.product_type_name, "Other")
d = d.merge(a, on="article_id", how="left")
d = d.rename(columns={"sales_channel_id": "channel"}).sort_values(["article_id","channel","week"]).reset_index(drop=True)
g = d.groupby(["article_id","channel"])
d["series_median_price"] = g.price.transform("median")
d["list_price"] = g.price.transform(lambda s: s.quantile(.9))
d["log_price_rel"] = np.log(d.price / d.series_median_price)
d["discount_depth"] = (1 - d.price / d.list_price).clip(lower=0)
d["n_weeks"] = g.units.transform("size"); d["total_units"] = g.units.transform("sum")
d["price_range"] = g.price.transform("max") / g.price.transform("min") - 1
d["age_weeks"] = ((d.week - d.groupby("article_id").week.transform("min")).dt.days // 7)
d["log_units"] = np.log(d.units)
d["eligible"] = (d.n_weeks >= MIN_WEEKS) & (d.total_units >= MIN_UNITS) & (d.price_range >= MIN_PRICE_RANGE)
d.to_parquet(DATA / "panel.parquet", index=False)

s = d.drop_duplicates(["article_id","channel"])
print(f"series (article x channel): {len(s):,}; eligible {s.eligible.sum():,}; eligible articles {s[s.eligible].article_id.nunique():,}")
print("eligible rows:", int(d.eligible.sum()), "| share of units:", round(d[d.eligible].units.sum() / d.units.sum(), 3))
print("categories:", d.category.nunique())
print(d[d.eligible].groupby("category").agg(articles=("article_id","nunique"), rows=("units","size"), units=("units","sum")).sort_values("units", ascending=False).to_string())
print("price move |log ratio| in eligible rows: median %.3f, p90 %.3f" % (d[d.eligible].log_price_rel.abs().median(), d[d.eligible].log_price_rel.abs().quantile(.9)))
