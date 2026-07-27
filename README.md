# 🏎️ GlobalValuation AI: Multi-Market Automotive Valuation Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)

An end-to-end machine learning system and interactive analytics platform designed to predict pre-owned vehicle valuations across international markets (**United States 🇺🇸** and **India 🇮🇳**). 

Unlike traditional single-dataset implementations, **GlobalValuation AI** utilizes a **Dual-Pipeline Comparative Architecture**. It isolates regional string formats, currency dynamics, and market-specific features into domain-adapted preprocessing routines and distinct ensemble regression models (XGBoost / LightGBM) with user-facing Explainable AI (**SHAP**).

---

## 📌 Executive Summary & Architecture

Scraped auto listings present severe schema variance across geographic regions:
* **US Market (`$ / Miles`):** Focuses on engine displacement/horsepower (`355HP 5.3L V8`), vehicle title history (`clean_title`), and accident reports.
* **Indian Market (`₹ Lakhs / KMs`):** Focuses on owner count (`1st Owner`), engine power (`BHP / CC`), city location, and fuel mechanism.

Instead of concatenating mismatched features into a sparse, noisy matrix, GlobalValuation AI isolates the data pipelines:

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
│ Data: Taeef Najib CSV     │                             │ Data: Avika Kasliwal CSV  │
│ Script: src/clean_us.py   │                             │ Script: src/clean_in.py   │
│ Engine: Regex (HP, L)     │                             │ Engine: Regex (CC, BHP)   │
│ Target: Price ($ USD)     │                             │ Target: Price (Lakhs INR) │
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
