# 🏎️ GlobalValuation AI: Multi-Market Automotive Valuation Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)


An end-to-end machine learning platform and interactive web application designed to predict pre-owned vehicle market valuations across international regions (**United States 🇺🇸** and **India 🇮🇳**). 

Unlike typical single-dataset implementations, **GlobalValuation AI** utilizes a **Dual-Pipeline Comparative Machine Learning Architecture**. It handles regional noise, currency variations, and domain-specific schemas by training isolated preprocessing routines and distinct ensemble regression models (XGBoost / LightGBM) paired with user-facing Explainable AI (**SHAP**).

---

## 📌 Architectural Blueprint

Scraped auto listings display significant schema and volume variations across regional markets:
* **US Craigslist Market (`$ USD / Miles`):** Large volume (~420k listings) with outlier noise ($0, $1, or extreme listing prices), categorical cylinder specs (`6 cylinders`), drive configurations (`4wd/fwd/rwd`), body types (`sedan/truck/SUV`), and title statuses (`clean`).
* **Indian CarDekho Market (`₹ INR / Kilometers`):** ~15.4k listings tracking owner history (`First Owner`, `Second Owner`), seller classifications (`Individual`, `Dealer`), engine displacement (`CC`), and maximum power (`BHP`).

Instead of combining mismatched columns into a sparse matrix, GlobalValuation AI isolates the training pathways:

```text
               ┌─────────────────────────────────────────────────┐
               │              GlobalValuation AI                 │
               └────────────────────────┬────────────────────────┘
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
┌───────────────────────────┐                             ┌───────────────────────────┐
│     US MARKET ROUTE       │                             │   INDIAN MARKET ROUTE     │
├───────────────────────────┤                             ├───────────────────────────┤
│ Data: Austin Reese CSV    │                             │ Data: Manish Kumar CSV    │
│ (~420k Craigslist rows)   │                             │ (~15.4k CarDekho rows)    │
│ Script: src/clean_us.py   │                             │ Script: src/clean_in.py   │
│ Engine: Outlier & Cyls    │                             │ Engine: CC, BHP, Owner    │
│ Target: Price ($ USD)     │                             │ Target: Price (₹ INR)     │
│ Model: us_pipeline.joblib │                             │ Model: in_pipeline.joblib │
└──────────┬────────────────┘                             └──────────┬────────────────┘
           │                                                         │
           └────────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                   ┌────────────────────────────────────────┐
                   │          Streamlit Web App             │
                   │     (Dynamic Regional Toggle)          │
                   └────────────────────────────────────────┘
