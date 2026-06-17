
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.agents import run_pipeline

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI",
    page_icon="💰",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .risk-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 20px;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 8px 0;
    }
    .risk-low    { background-color: #d4edda; color: #155724; }
    .risk-medium { background-color: #fff3cd; color: #856404; }
    .risk-high   { background-color: #f8d7da; color: #721c24; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────
st.title("💰 FinSight AI")
st.caption("Upload your UPI statement · Get AI-powered spending insights")
st.divider()


# ── Sidebar inputs ───────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload & Settings")

    pdf_file = st.file_uploader(
        "Upload UPI Statement PDF",
        type=["pdf"],
        help="Supports Google Pay and PhonePe statements"
    )

    monthly_income = st.number_input(
        "Monthly Income / Pocket Money (₹)",
        min_value=1000,
        max_value=500000,
        value=15000,
        step=500,
        help="Your total monthly take-home income"
    )

    analyse_btn = st.button("🔍 Analyse My Spending", use_container_width=True, type="primary")

    st.divider()
    st.caption("FinSight AI uses DistilBERT + Groq LLaMA to categorise your transactions and generate personalised suggestions.")


# ── Main logic ───────────────────────────────────────────────
if analyse_btn and pdf_file:
    with st.spinner("Running analysis pipeline..."):
        result = run_pipeline(pdf_file, monthly_income)

    if result["error"]:
        st.error(f"Something went wrong: {result['error']}")
        st.stop()

    # Store in session state
    st.session_state["result"] = result


# ── Results ──────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    df     = pd.DataFrame(result["categorized_df"])
    r      = result["risk_report"]

    # ── Row 1: Key metrics ────────────────────────────────────
    st.subheader("📊 Spending Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Spent", f"₹{r['total_spent']:,.2f}")
    with col2:
        st.metric("Monthly Income", f"₹{r['monthly_income']:,.0f}")
    with col3:
        st.metric("Savings Left", f"₹{r['savings_potential']:,.2f}")
    with col4:
        st.metric("Transactions", r["transaction_count"])

    # ── Risk badge ────────────────────────────────────────────
    st.subheader("🎯 Risk Assessment")
    risk = r["risk_level"]
    badge_class = f"risk-{risk.lower()}"
    icons = {"Low": "✅", "Medium": "⚠️", "High": "🔴"}
    st.markdown(
        f'<div class="risk-badge {badge_class}">{icons.get(risk,"⚠️")} {risk} Risk — {r["risk_score"]}% of income spent</div>',
        unsafe_allow_html=True
    )

    # Risk progress bar
    st.progress(min(r["spend_ratio"], 1.0))
    st.caption(f"Recommended maximum: 75% of income on expenses")

    st.divider()

    # ── Row 2: Charts ─────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🍕 Spending by Category")
        cat_df = pd.DataFrame(r["category_summary"])
        if not cat_df.empty:
            fig_pie = px.pie(
                cat_df,
                values="total",
                names="category",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("📈 Category vs Income Limit")
        if not cat_df.empty:
            from src.risk import CATEGORY_LIMITS
            cat_df["limit_pct"] = cat_df["category"].map(
                lambda c: CATEGORY_LIMITS.get(c, 0.15) * 100
            )
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Spent %",
                x=cat_df["category"],
                y=cat_df["percentage"],
                marker_color="#4C9BE8"
            ))
            fig_bar.add_trace(go.Bar(
                name="Limit %",
                x=cat_df["category"],
                y=cat_df["limit_pct"],
                marker_color="#FF6B6B",
                opacity=0.5
            ))
            fig_bar.update_layout(
                barmode="group",
                margin=dict(t=0,b=0,l=0,r=0),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Overspent categories ──────────────────────────────────
    if r["overspent"]:
        st.subheader("🚨 Overspent Categories")
        for o in r["overspent"]:
            with st.expander(f"⚠️ {o['category']} — spent {o['actual_pct']}% (limit {o['limit_pct']}%)"):
                st.write(f"**Amount spent:** ₹{o['spent']:,.2f}")
                st.write(f"**Excess over limit:** ₹{o['excess']:,.2f}")
                st.write(f"**Recommended max:** {o['limit_pct']}% = ₹{o['limit_pct']/100 * r['monthly_income']:,.0f}")
        st.divider()

    # ── AI Suggestions ────────────────────────────────────────
    st.subheader("🤖 AI Suggestions")
    st.markdown(result["suggestions"])
    st.divider()

    # ── Transaction table ─────────────────────────────────────
    st.subheader("📋 Transaction Details")

    # Filter controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_cats = st.multiselect(
            "Filter by category",
            options=df["category"].unique().tolist(),
            default=df["category"].unique().tolist()
        )
    with col_f2:
        txn_type = st.radio(
            "Transaction type",
            ["All", "Debit only", "Credit only"],
            horizontal=True
        )

    filtered_df = df[df["category"].isin(selected_cats)]
    if txn_type == "Debit only":
        filtered_df = filtered_df[filtered_df["transaction_type"] == "debit"]
    elif txn_type == "Credit only":
        filtered_df = filtered_df[filtered_df["transaction_type"] == "credit"]

    st.dataframe(
        filtered_df[["date","description","amount","category","transaction_type","method","confidence"]],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ── Download report ───────────────────────────────────────
    st.subheader("📥 Download Report")
    if st.button("📄 Generate PDF Report", use_container_width=True):
        with st.spinner("Building your report..."):
            from src.report import build_report
            pdf_bytes = build_report(
                risk_report        = r,
                suggestions        = result["suggestions"],
                categorized_df_records = result["categorized_df"]
            )
        st.download_button(
            label="⬇️ Download FinSight Report",
            data=pdf_bytes,
            file_name=f"finsight_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )