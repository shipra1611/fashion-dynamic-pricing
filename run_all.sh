#!/bin/sh
# Needs data/articles.csv and data/hm_weekly_channel.csv.gz (see README for how to produce them from the Kaggle H&M data)
set -e
mkdir -p data outputs powerbi
cd src
python 01_prepare.py
python 02_elasticity.py
python 03_demand_model.py
python 04_pricing_ab.py
python 05_exports_charts.py
