"""Streamlit frontend for the Store Review Insight Analyzer."""
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analyzer import AnalyzerError, Settings, analyze
from prompts import build_review_prompt

load_dotenv()

SAMPLE_REVIEWS = """Fast delivery and the jacket fits perfectly!
The battery died after two days and support never answered my emails.
Box arrived on time. The product is fine, nothing special.
Great quality for the price, I would buy again.
Sizing chart is wrong, the medium is closer to a small.
Customer service was friendly and solved my refund in one day.
Delivery took three weeks and the tracking page was useless.
The charger stopped working after a month. Very disappointed.
Colors look different from the pictures but the material is nice.
Packaging was damaged but the item itself works well.
Checkout kept crashing on my phone, had to order from the laptop.
Love the design, the stitching feels solid and durable."""

st.set_page_config(page_title="Store Review Insight Analyzer", page_icon="🛍️", layout="wide")
st.title("🛍️ Store Review Insight Analyzer")
st.caption("Turn raw customer reviews into sentiment, topics, clusters and an executive report.")

# ---------- Sidebar: parameters ----------
with st.sidebar:
    st.header("Model settings")
    api_key = st.text_input("Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password")
    model = st.selectbox("Model", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05,
                            help="Lower = more consistent. Higher = more varied.")
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    max_tokens = st.number_input("Max tokens", min_value=200, max_value=2000, value=1500, step=50)
    clusters = st.slider("Number of clusters", 2, 8, 4)
    language = st.selectbox("Report language", ["English", "Hungarian"])

# ---------- Input ----------
source = st.radio("Input source", ["Paste reviews", "Upload CSV"], horizontal=True)
reviews = []

if source == "Paste reviews":
    st.button("Load sample reviews",
              on_click=lambda: st.session_state.update(review_text=SAMPLE_REVIEWS))
    text = st.text_area("One review per line", height=220, key="review_text",
                        placeholder="Paste reviews here, one per line...")
    reviews = text.splitlines()
else:
    uploaded = st.file_uploader("CSV file with a 'review' column", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "review" not in df.columns:
            st.error("The CSV must contain a column named 'review'.")
            st.stop()
        reviews = df["review"].dropna().astype(str).tolist()
        st.info(f"Loaded {len(reviews)} reviews from the file.")

with st.expander("🔍 Show the classification prompt (for the presentation)"):
    st.code(build_review_prompt("The package arrived late but the product is great."), language="text")

# ---------- Run ----------
if st.button("Analyze reviews", type="primary"):
    settings = Settings(
        model=model, temperature=temperature, top_p=top_p,
        max_tokens=int(max_tokens), clusters=clusters, language=language,
    )
    bar = st.progress(0.0, text="Starting analysis...")
    try:
        with st.spinner("Calling the model..."):
            result = analyze(reviews, settings, api_key=api_key,
                             progress=lambda p: bar.progress(p, text=f"Processing... {int(p * 100)}%"))
    except AnalyzerError as e:
        st.error(str(e))
        st.stop()
    bar.empty()

    st.session_state["result"] = result

# ---------- Display ----------
result = st.session_state.get("result")
if result:
    rows = result["results"]
    total = len(rows)
    counts = pd.Series([r["sentiment"] for r in rows]).value_counts()
    pct = lambda k: f"{counts.get(k, 0) / total:.0%}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews analyzed", total)
    c2.metric("Positive", pct("positive"))
    c3.metric("Neutral", pct("neutral"))
    c4.metric("Negative", pct("negative"))

    left, right = st.columns(2)
    with left:
        st.subheader("Most frequent topics")
        topics = pd.Series(result["topic_counts"]).sort_values(ascending=False).head(10)
        st.bar_chart(topics)
    with right:
        st.subheader("Thematic clusters")
        cluster_df = pd.DataFrame(
            [{"Cluster": c["label"], "Reviews": c["size"]} for c in result["clusters"].values()]
        ).sort_values("Reviews", ascending=False)
        st.dataframe(cluster_df, hide_index=True, use_container_width=True)

    st.subheader("📝 Executive report")
    st.markdown(result["report"])

    with st.expander("Per-review results"):
        df_out = pd.DataFrame(rows)
        df_out["topics"] = df_out["topics"].apply(", ".join)
        df_out["cluster"] = [
            next((c["label"] for c in result["clusters"].values() if i in c["reviews"]), "")
            for i in range(total)
        ]
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Download results (CSV)", df_out.to_csv(index=False).encode("utf-8"),
                           file_name="review_analysis.csv", mime="text/csv")