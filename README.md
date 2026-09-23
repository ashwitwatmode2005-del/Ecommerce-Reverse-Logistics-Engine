# 🔄 E-Commerce Reverse Logistics & Return-Prediction Decision Engine

## Problem Statement

E-commerce platforms face significant financial and operational pressure from product returns. High return rates — driven by factors such as Cash-on-Delivery (COD) misuse, size mismatches, and logistics failures — erode profit margins, inflate reverse logistics freight costs, and create avoidable environmental waste.

This project builds an end-to-end **Reverse Logistics Decision Engine** that:
- Provides executives with a real-time KPI dashboard of return rates, lost GMV, and freight costs.
- Diagnoses the root causes of returns across payment methods, product categories, return reasons, and geographic clusters.
- Deploys an interactive **ML-powered Return Risk Scoring Engine** that predicts the probability of return for any new order and recommends pre-dispatch business interventions for high-risk orders.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI / App Framework | Streamlit |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-Learn (RandomForestClassifier) |
| Visualisation | Plotly Express & Graph Objects |

---

## Project Structure

```
ecommerce_reverse_logistics/
├── app.py               # Full Streamlit application (preprocessing + ML + UI)
├── returns_data.csv     # Source dataset
├── requirements.txt     # Pinned Python dependencies
└── README.md            # This file
```

---

## Setup & Execution

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Dashboard Overview

### Tab 1 — Executive KPI Overview
- **Metric cards**: Total Orders, Overall Return Rate (%), Total Lost GMV, Total Reverse Logistics Freight Cost.
- **Monthly time-series chart**: Order Volume vs. Return Volume.
- **Donut chart**: Return Status distribution.

### Tab 2 — Root-Cause Driver Diagnostics
- Return Rate (%) by **Payment Method** (COD highlighted vs. digital modes).
- Return Rate (%) by **Product Category** (horizontal bar).
- **Top Return Reasons** distribution.
- **Return Volume by User Location** (top 15 geographic clusters).

### Tab 3 — Return Risk Decision Engine
- RandomForestClassifier (200 trees) trained on 9 order-level features.
- Interactive form to enter order details and get an **instant Return Probability (%)**.
- **High Risk (>60%)**: Recommends digital advance deposit, OTP address verification, and automated size/spec check prompt.
- **Low/Medium Risk (≤60%)**: Approves for standard automated warehouse fulfillment.

---

## Global Sidebar Filters

The sidebar provides dynamic filters applied to Tab 1 and Tab 2:
- **Product Category** (multi-select)
- **Payment Method** (multi-select)
- **User Location** (multi-select)
- **Order Date Range** (date picker)

Tab 3 always uses the full dataset for model training, independent of sidebar filters.

---

## Dataset Source

**Kaggle Dataset URL:** `https://www.kaggle.com/datasets/sowmihari/returns-management`

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | RandomForestClassifier |
| Estimators | 200 |
| Train/Test Split | 80% / 20% |
| Target Variable | `Return_Status` (1 = Returned, 0 = Not Returned) |
| Categorical Encoding | OneHotEncoder (handle_unknown="ignore") |
| Data Leakage Prevention | Dropped: Order_ID, Product_ID, User_ID, Order_Date, Return_Date, Return_Reason, Days_to_Return, Return_Cost, Profit_Loss, CO2_Saved, Waste_Avoided |

---
