"""
dashboard.py — Customer Feedback Intelligence Dashboard (Streamlit)

Run with:  streamlit run dashboard.py
Or via:    make run
"""

import sys
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nlp_pipeline import run_full_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).resolve().parent / "data" / "reviews.csv"


@st.cache_resource
def load_results():
    results = run_full_pipeline(str(DATA_PATH))
    return results


results = load_results()
df = results["data"]
topic_df = results["topics"]
trends = results["trends"]
recommendations = results["recommendations"]
summary = results["summary"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Feedback Intel")
st.sidebar.markdown("**Customer Feedback Intelligence System**")
st.sidebar.markdown("---")

st.sidebar.subheader("Summary Metrics")
col1, col2, col3 = st.sidebar.columns(3)
col1.metric("Reviews", summary["total_reviews"])
col2.metric("Avg Rating", summary["avg_rating"])
col3.metric("Topics", summary["topics_found"])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

category_filter = st.sidebar.multiselect(
    "Product Category",
    options=df["product_category"].unique().tolist(),
    default=df["product_category"].unique().tolist(),
)

sentiment_filter = st.sidebar.multiselect(
    "Sentiment",
    options=["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"],
)

rating_range = st.sidebar.slider(
    "Rating Range", 1, 5, (1, 5)
)

# Apply filters
mask = (
    df["product_category"].isin(category_filter)
    & df["sentiment"].isin(sentiment_filter)
    & df["rating"].between(rating_range[0], rating_range[1])
)
filtered_df = df[mask]

# ── Main content ──────────────────────────────────────────────────────────────
st.title("📈 Customer Feedback Intelligence System")
st.markdown(
    "An NLP-powered dashboard that extracts sentiment, topics, and trends from "
    "customer reviews to generate actionable business insights."
)

# ── Row 1: Sentiment KPIs ─────────────────────────────────────────────────────
st.subheader("🎯 Sentiment Overview")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
pos_pct = (filtered_df["sentiment"] == "Positive").mean() * 100
neu_pct = (filtered_df["sentiment"] == "Neutral").mean() * 100
neg_pct = (filtered_df["sentiment"] == "Negative").mean() * 100

kpi1.metric("😊 Positive", f"{pos_pct:.1f}%")
kpi2.metric("😐 Neutral", f"{neu_pct:.1f}%")
kpi3.metric("☹️ Negative", f"{neg_pct:.1f}%")
kpi4.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f}")

# ── Row 2: Charts ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment Distribution")
    sent_counts = filtered_df["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["sentiment", "count"]
    colors = {"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"}
    fig_pie = px.pie(
        sent_counts,
        values="count",
        names="sentiment",
        color="sentiment",
        color_discrete_map=colors,
        hole=0.4,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Rating Distribution")
    rating_dist = filtered_df["rating"].value_counts().sort_index().reset_index()
    rating_dist.columns = ["rating", "count"]
    fig_hist = px.bar(
        rating_dist,
        x="rating",
        y="count",
        color="rating",
        color_continuous_scale="RdYlGn",
        text_auto=True,
    )
    fig_hist.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Row 3: Sentiment by Category ──────────────────────────────────────────────
st.subheader("🏷️ Sentiment by Product Category")
cat_sent = (
    filtered_df.groupby(["product_category", "sentiment"])
    .size()
    .reset_index(name="count")
)
fig_cat = px.bar(
    cat_sent,
    x="product_category",
    y="count",
    color="sentiment",
    color_discrete_map=colors,
    barmode="group",
    text_auto=True,
)
st.plotly_chart(fig_cat, use_container_width=True)

# ── Row 4: Trends ─────────────────────────────────────────────────────────────
st.subheader("📅 Monthly Trends")
trend_left, trend_right = st.columns(2)

with trend_left:
    rating_trend = trends["rating_trend"]
    fig_rt = px.line(
        rating_trend,
        x="month",
        y="avg_rating",
        markers=True,
        title="Average Rating Over Time",
        labels={"avg_rating": "Avg Rating", "month": "Month"},
        range_y=[1, 5],
    )
    fig_rt.add_bar(
        x=rating_trend["month"],
        y=rating_trend["review_count"],
        name="Review Count",
        yaxis="y2",
        marker_color="rgba(100, 149, 237, 0.3)",
    )
    fig_rt.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="Review Count", showgrid=False),
    )
    st.plotly_chart(fig_rt, use_container_width=True)

with trend_right:
    sent_trend = trends["sentiment_trend"]
    sent_melted = sent_trend.melt(id_vars="month", var_name="sentiment", value_name="count")
    fig_st = px.area(
        sent_melted,
        x="month",
        y="count",
        color="sentiment",
        color_discrete_map=colors,
        title="Sentiment Volume Over Time",
        labels={"count": "Count", "month": "Month"},
    )
    st.plotly_chart(fig_st, use_container_width=True)

# ── Row 5: Topics ─────────────────────────────────────────────────────────────
st.subheader("🔍 Extracted Topics")
for _, row in topic_df.iterrows():
    with st.expander(f"**{row['topic']}**"):
        st.markdown(f"**Keywords:** {row['keywords']}")

# Topic-reviews mapping
st.subheader("📄 Topic Assignment (Sample)")
topic_sample = (
    filtered_df[["review_text", "rating", "sentiment", "topic_id"]]
    .head(10)
    .copy()
)
topic_sample["topic_id"] = topic_sample["topic_id"].apply(lambda x: f"Topic {x}")
st.dataframe(topic_sample, use_container_width=True)

# ── Row 6: Recommendations ────────────────────────────────────────────────────
st.subheader("💡 Action Recommendations")

if recommendations:
    rec_df = pd.DataFrame(recommendations)
    # Color-code urgency
    urgency_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    rec_df["urgency"] = rec_df["urgency"].map(urgency_colors)

    for _, rec in pd.DataFrame(recommendations).iterrows():
        icon = urgency_colors.get(rec["urgency"], "⚪")
        with st.container(border=True):
            col_a, col_b = st.columns([1, 5])
            with col_a:
                st.markdown(f"### {icon}")
                st.caption(rec["urgency"])
            with col_b:
                st.markdown(f"**{rec['area']}**")
                st.markdown(f"*{rec['finding']}*")
                st.markdown(f"👉 {rec['action']}")
else:
    st.info("No specific recommendations — sentiment is well-balanced.")

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("📁 Raw Data Preview"):
    st.dataframe(filtered_df, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Streamlit · VADER Sentiment · TF-IDF + NMF Topic Modeling · Plotly"
)
