import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Reverse Logistics Engine",
    page_icon="🔄",
    layout="wide",
)

# ─────────────────────────────────────────────
# Data loading & preprocessing
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("returns_data.csv", parse_dates=["Order_Date"])
    df["Return_Binary"] = (df["Return_Status"] == "Returned").astype(int)
    return df


df_full = load_data()

# ─────────────────────────────────────────────
# ML model training (uses full dataset, no sidebar filters)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "Product_Category", "Product_Price", "Order_Quantity",
    "Discount_Applied", "Payment_Method", "Shipping_Method",
    "User_Age", "User_Gender", "User_Location",
]
CAT_FEATURES = ["Product_Category", "Payment_Method", "Shipping_Method", "User_Gender", "User_Location"]
NUM_FEATURES = ["Product_Price", "Order_Quantity", "Discount_Applied", "User_Age"]


@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[FEATURE_COLS].copy()
    y = data["Return_Binary"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    report = classification_report(y_test, pipeline.predict(X_test), output_dict=True)
    return pipeline, report


model_pipeline, model_report = train_model(df_full)

# ─────────────────────────────────────────────
# Sidebar – Global Dynamic Controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔄 Reverse Logistics")
    st.divider()

    all_categories = sorted(df_full["Product_Category"].dropna().unique().tolist())
    all_payments   = sorted(df_full["Payment_Method"].dropna().unique().tolist())
    all_locations  = sorted(df_full["User_Location"].dropna().unique().tolist())

    sel_categories = st.multiselect("Product Category", all_categories, default=[])
    sel_payments   = st.multiselect("Payment Method",   all_payments,   default=[])
    sel_locations  = st.multiselect("User Location",    all_locations,  default=[])

    min_date = df_full["Order_Date"].min().date()
    max_date = df_full["Order_Date"].max().date()
    date_range = st.date_input(
        "Order Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.divider()


# ─────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────
df = df_full.copy()

if sel_categories:
    df = df[df["Product_Category"].isin(sel_categories)]
if sel_payments:
    df = df[df["Payment_Method"].isin(sel_payments)]
if sel_locations:
    df = df[df["User_Location"].isin(sel_locations)]

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df = df[(df["Order_Date"] >= start_date) & (df["Order_Date"] <= end_date)]

# ─────────────────────────────────────────────
# App header
# ─────────────────────────────────────────────
st.title("🔄 E-Commerce Reverse Logistics & Return-Prediction Engine")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "📊 Executive KPI Overview",
    "🔍 Root-Cause Driver Diagnostics",
    "🤖 Return Risk Decision Engine",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Executive KPI Overview
# ═══════════════════════════════════════════════════════════════
with tab1:
    if df.empty:
        st.warning("No data matches the current filter selection.")
    else:
        total_orders   = len(df)
        returned_df    = df[df["Return_Binary"] == 1]
        return_rate    = (len(returned_df) / total_orders * 100) if total_orders else 0
        lost_gmv       = (returned_df["Product_Price"] * returned_df["Order_Quantity"]).sum()
        freight_cost   = returned_df["Return_Cost"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🛒 Total Orders",          f"{total_orders:,}")
        c2.metric("↩️ Overall Return Rate",    f"{return_rate:.1f}%")
        c3.metric("💸 Lost GMV",              f"₹{lost_gmv:,.0f}")
        c4.metric("🚚 Reverse Freight Cost",  f"₹{freight_cost:,.0f}")

        st.divider()

        # Monthly Order vs Return Volume
        monthly = (
            df.set_index("Order_Date")
            .resample("ME")
            .agg(
                Order_Volume=("Order_ID", "count"),
                Return_Volume=("Return_Binary", "sum"),
            )
            .reset_index()
        )
        monthly["Month"] = monthly["Order_Date"].dt.strftime("%b %Y")

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Order_Volume"],
            mode="lines+markers", name="Order Volume",
            line=dict(color="#3b82f6", width=2),
        ))
        fig_ts.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Return_Volume"],
            mode="lines+markers", name="Return Volume",
            line=dict(color="#ef4444", width=2, dash="dash"),
        ))
        fig_ts.update_layout(
            title="Monthly Order Volume vs. Return Volume",
            xaxis_title="Month", yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
            height=400,
        )

        # Donut chart – Return Status distribution
        status_counts = df["Return_Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig_donut = px.pie(
            status_counts, names="Status", values="Count",
            hole=0.55,
            color_discrete_sequence=["#ef4444", "#22c55e"],
            title="Return Status Distribution",
        )
        fig_donut.update_traces(textposition="outside", textinfo="percent+label")
        fig_donut.update_layout(height=400, showlegend=True)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.plotly_chart(fig_ts, use_container_width=True)
        with col_right:
            st.plotly_chart(fig_donut, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Root-Cause Driver Diagnostics
# ═══════════════════════════════════════════════════════════════
with tab2:
    if df.empty:
        st.warning("No data matches the current filter selection.")
    else:
        # Return Rate by Payment Method
        pay_stats = (
            df.groupby("Payment_Method")
            .agg(Total=("Order_ID", "count"), Returned=("Return_Binary", "sum"))
            .reset_index()
        )
        pay_stats["Return_Rate"] = pay_stats["Returned"] / pay_stats["Total"] * 100
        pay_stats["Color"] = pay_stats["Payment_Method"].apply(
            lambda x: "#ef4444" if x == "COD" else "#3b82f6"
        )
        fig_pay = px.bar(
            pay_stats.sort_values("Return_Rate", ascending=False),
            x="Payment_Method", y="Return_Rate",
            color="Payment_Method",
            color_discrete_map={m: ("#ef4444" if m == "COD" else "#3b82f6") for m in pay_stats["Payment_Method"]},
            title="Return Rate (%) by Payment Method — COD vs. Digital",
            labels={"Return_Rate": "Return Rate (%)", "Payment_Method": "Payment Method"},
            text_auto=".1f",
        )
        fig_pay.update_layout(height=380, showlegend=False)

        # Return Rate by Product Category (horizontal)
        cat_stats = (
            df.groupby("Product_Category")
            .agg(Total=("Order_ID", "count"), Returned=("Return_Binary", "sum"))
            .reset_index()
        )
        cat_stats["Return_Rate"] = cat_stats["Returned"] / cat_stats["Total"] * 100
        fig_cat = px.bar(
            cat_stats.sort_values("Return_Rate"),
            y="Product_Category", x="Return_Rate",
            orientation="h",
            title="Return Rate (%) by Product Category",
            labels={"Return_Rate": "Return Rate (%)", "Product_Category": "Category"},
            color="Return_Rate",
            color_continuous_scale="Reds",
            text_auto=".1f",
        )
        fig_cat.update_layout(height=400, coloraxis_showscale=False)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pay, use_container_width=True)
        with col2:
            st.plotly_chart(fig_cat, use_container_width=True)

        st.divider()

        # Top Return Reasons
        reason_df = (
            df[df["Return_Binary"] == 1]["Return_Reason"]
            .value_counts()
            .reset_index()
        )
        reason_df.columns = ["Return_Reason", "Count"]
        reason_df = reason_df[reason_df["Return_Reason"] != "No Return"]
        fig_reason = px.bar(
            reason_df,
            x="Return_Reason", y="Count",
            title="Top Return Reason Categories",
            labels={"Count": "Number of Returns", "Return_Reason": "Reason"},
            color="Count",
            color_continuous_scale="Oranges",
            text_auto=True,
        )
        fig_reason.update_layout(height=380, coloraxis_showscale=False)

        # Return Volume by User Location (top 15)
        loc_df = (
            df[df["Return_Binary"] == 1]["User_Location"]
            .value_counts()
            .head(15)
            .reset_index()
        )
        loc_df.columns = ["User_Location", "Return_Count"]
        fig_loc = px.bar(
            loc_df.sort_values("Return_Count"),
            y="User_Location", x="Return_Count",
            orientation="h",
            title="Return Volume by User Location (Top 15)",
            labels={"Return_Count": "Return Volume", "User_Location": "Location"},
            color="Return_Count",
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_loc.update_layout(height=420, coloraxis_showscale=False)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(fig_reason, use_container_width=True)
        with col4:
            st.plotly_chart(fig_loc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Interactive Return Risk Decision Engine
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🤖 Return Risk Decision Engine")
    st.caption(
        "Model: RandomForestClassifier (200 trees) trained on the full dataset. "
        "Sidebar filters do **not** affect model training."
    )

    # Model performance summary
    with st.expander("📈 Model Performance Summary"):
        acc = model_report.get("accuracy", 0)
        prec = model_report.get("1", {}).get("precision", 0)
        rec  = model_report.get("1", {}).get("recall", 0)
        f1   = model_report.get("1", {}).get("f1-score", 0)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy",  f"{acc*100:.1f}%")
        mc2.metric("Precision (Returned)", f"{prec*100:.1f}%")
        mc3.metric("Recall (Returned)",    f"{rec*100:.1f}%")
        mc4.metric("F1-Score (Returned)",  f"{f1*100:.1f}%")

    st.divider()
    st.markdown("### 📋 Order Risk Evaluation Form")
    st.caption("Fill in the order details below and click **Evaluate Order Risk**.")

    uniq = lambda col: sorted(df_full[col].dropna().unique().tolist())

    with st.form("risk_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            inp_category  = st.selectbox("Product Category",  uniq("Product_Category"))
            inp_payment   = st.selectbox("Payment Method",    uniq("Payment_Method"))
            inp_shipping  = st.selectbox("Shipping Method",   uniq("Shipping_Method"))
        with r1c2:
            inp_price     = st.number_input("Product Price (₹)",   min_value=0.0,  max_value=50000.0, value=1000.0, step=50.0)
            inp_quantity  = st.number_input("Order Quantity",       min_value=1,    max_value=50,      value=1, step=1)
            inp_discount  = st.number_input("Discount Applied (%)", min_value=0.0,  max_value=100.0,   value=10.0, step=0.5)
        with r1c3:
            inp_age       = st.slider("User Age",          min_value=18, max_value=80, value=30)
            inp_gender    = st.selectbox("User Gender",    uniq("User_Gender"))
            inp_location  = st.selectbox("User Location",  uniq("User_Location"))

        submitted = st.form_submit_button("🔍 Evaluate Order Risk", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Product_Category": inp_category,
            "Product_Price":    inp_price,
            "Order_Quantity":   inp_quantity,
            "Discount_Applied": inp_discount,
            "Payment_Method":   inp_payment,
            "Shipping_Method":  inp_shipping,
            "User_Age":         inp_age,
            "User_Gender":      inp_gender,
            "User_Location":    inp_location,
        }])

        proba = model_pipeline.predict_proba(input_df)[0][1]
        risk_pct = proba * 100

        st.divider()
        st.markdown("### 📊 Risk Assessment Result")

        col_gauge, col_action = st.columns([2, 3])
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "Return Probability", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if risk_pct > 60 else "#22c55e"},
                    "steps": [
                        {"range": [0,  40],  "color": "#dcfce7"},
                        {"range": [40, 60],  "color": "#fef9c3"},
                        {"range": [60, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#1f2328", "width": 3},
                        "thickness": 0.75,
                        "value": 60,
                    },
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_action:
            if risk_pct > 60:
                st.error(f"⚠️ **HIGH RISK ORDER — Return Probability: {risk_pct:.1f}%**")
                st.markdown("""
**Recommended Pre-Dispatch Business Actions:**

1. 💳 **Require Digital Advance Deposit** — Block Cash-on-Delivery for this order; mandate prepayment via UPI, card, or wallet to reduce COD-driven return abuse.
2. 📍 **OTP Address Verification** — Trigger an automated OTP to the delivery address before dispatching to confirm address validity and buyer intent.
3. 📐 **Automated Customer Size / Spec Check Prompt** — Send a personalised size-guide or product specification prompt via email/SMS to pre-empt fit/spec mismatch returns.
4. 🔎 **Fraud & Abuse Flag** — Route order to manual review queue if the same User_ID shows repeated high-risk patterns.
                """)
            else:
                st.success(f"✅ **LOW / MEDIUM RISK — Return Probability: {risk_pct:.1f}%**")
                st.markdown("""
**Recommended Action:**

✅ **Approve for Standard Automated Warehouse Fulfillment.**

- No additional intervention required.
- Standard SLA and carrier assignment applies.
- Monitor post-delivery feedback for any emerging signals.
                """)

        # Input summary table
        with st.expander("📄 Input Summary"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
