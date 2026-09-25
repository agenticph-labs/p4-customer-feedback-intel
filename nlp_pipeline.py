"""
nlp_pipeline.py — NLP Analysis Pipeline for Customer Feedback

Provides sentiment analysis, topic extraction, and trend detection
for customer reviews using VADER, TF-IDF + NMF, and temporal aggregation.
"""

import re
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Import VADER ──────────────────────────────────────────────────────────────
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk

    nltk.download("vader_lexicon", quiet=True)
    _sia = SentimentIntensityAnalyzer()
except Exception:
    _sia = None

# ── Stopwords ─────────────────────────────────────────────────────────────────
_STOP_WORDS = set(
    "a an the and or but in on at to for of is was are were it its be has have "
    "do does did will would can could shall should may might am been being "
    "having doing getting got get go goes going come came comes coming take "
    "takes took taking make made makes making use used uses using know knows "
    "knew knowing see saw seen sees seeing think thinks thought thinking "
    "want wants wanted wanting look looks looked looking need needs needed "
    "needing seem seems seemed seeming buy bought buying sells sold selling "
    "just also very really much many such like even still too well back "
    "now then here there this that these those i we you he she they my our "
    "your his her their me us him them mine yours hers its ours theirs "
    "what which who whom this that these those about into during before "
    "after above below between out off over under again further once "
    "all each every both few more most other some any no none not only "
    "own same so than as because if while although though how when where why"
    .split()
)


def preprocess(text: str) -> str:
    """Lowercase, remove non-alphabetic tokens, strip stopwords."""
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return " ".join(t for t in tokens if t not in _STOP_WORDS and len(t) > 2)


def classify_sentiment(compound: float) -> str:
    """Map VADER compound score → categorical label."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def compute_sentiment(texts: list[str]) -> pd.DataFrame:
    """Return DataFrame with compound score and label per review."""
    results: list[dict[str, Any]] = []
    for text in texts:
        score = _sia.polarity_scores(text) if _sia else {"compound": 0.0}
        results.append(
            {
                "compound": score["compound"],
                "sentiment": classify_sentiment(score["compound"]),
                "pos": score.get("pos", 0),
                "neu": score.get("neu", 0),
                "neg": score.get("neg", 0),
            }
        )
    return pd.DataFrame(results)


def extract_topics(
    texts: list[str],
    n_topics: int = 5,
    n_top_words: int = 8,
) -> tuple[pd.DataFrame, Any, Any]:
    """Extract topics via TF-IDF → NMF.

    Returns (topic_df, vectorizer, nmf_model).
    """
    processed = [preprocess(t) for t in texts]
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, max_features=1500)
    tfidf = vectorizer.fit_transform(processed)
    n_topics = min(n_topics, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    n_topics = max(n_topics, 2)

    nmf = NMF(n_components=n_topics, random_state=42, init="random", max_iter=400, l1_ratio=0.5)
    nmf.fit(tfidf)
    feature_names = vectorizer.get_feature_names_out()

    rows = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_features = [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
        rows.append({"topic": f"Topic {topic_idx + 1}", "keywords": ", ".join(top_features)})

    return pd.DataFrame(rows), vectorizer, nmf


def assign_dominant_topics(
    texts: list[str],
    vectorizer: Any,
    nmf_model: Any,
) -> pd.Series:
    """Assign each review to its dominant NMF topic."""
    processed = [preprocess(t) for t in texts]
    tfidf = vectorizer.transform(processed)
    topic_weights = nmf_model.transform(tfidf)
    return pd.Series(topic_weights.argmax(axis=1) + 1, name="topic_id")


def detect_trends(
    df: pd.DataFrame,
    date_col: str = "date",
    rating_col: str = "rating",
    sentiment_col: str = "sentiment",
) -> dict[str, Any]:
    """Monthly-aggregated trend data for ratings and sentiment."""
    temp = df.copy()
    temp["month"] = pd.to_datetime(temp[date_col]).dt.to_period("M").astype(str)

    rating_trend = temp.groupby("month")[rating_col].agg(["mean", "count"]).reset_index()
    rating_trend.columns = ["month", "avg_rating", "review_count"]

    sent_trend = (
        temp.groupby(["month", sentiment_col])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    return {
        "rating_trend": rating_trend,
        "sentiment_trend": sent_trend,
        "total_reviews": len(df),
        "months_covered": sorted(temp["month"].unique().tolist()),
    }


def generate_recommendations(
    df: pd.DataFrame,
    topic_df: pd.DataFrame,
) -> list[dict[str, str]]:
    """Derive action recommendations from data patterns."""
    recs: list[dict[str, str]] = []

    # 1. Low-rated topics → complaints
    low_rated = df[df["rating"] <= 2]
    if not low_rated.empty:
        low_text = " ".join(preprocess(t) for t in low_rated["review_text"])
        # extract frequent complaint keywords from low-rated reviews
        word_freq = pd.Series(low_text.split()).value_counts().head(10)
        complaint_areas = ", ".join(word_freq.index[:5])
        recs.append(
            {
                "area": "Product Quality Issues",
                "finding": f"Top complaint keywords: {complaint_areas}",
                "action": "Initiate root-cause analysis on mentioned components; escalate quality checks.",
                "urgency": "High" if len(low_rated) > 15 else "Medium",
            }
        )

    # 2. Negative sentiment by category
    neg_by_cat = (
        df[df["sentiment"] == "Negative"]
        .groupby("product_category")
        .size()
        .sort_values(ascending=False)
    )
    if not neg_by_cat.empty:
        worst_cat = neg_by_cat.index[0]
        recs.append(
            {
                "area": f"Negative Sentiment — {worst_cat}",
                "finding": f"{neg_by_cat.iloc[0]} negative reviews in {worst_cat}, highest of all categories.",
                "action": f"Review {worst_cat.lower()} QA process and customer support script for this category.",
                "urgency": "High",
            }
        )

    # 3. Shipping / delivery / returns
    shipping_words = ["shipping", "delivery", "return", "arrived", "packaging", "damaged"]
    shipping_reviews = df[
        df["review_text"].str.lower().apply(
            lambda t: any(w in t for w in shipping_words)
        )
    ]
    if not shipping_reviews.empty and len(shipping_reviews) >= 3:
        recs.append(
            {
                "area": "Shipping & Fulfillment",
                "finding": f"{len(shipping_reviews)} reviews mention shipping, delivery, or returns.",
                "action": "Audit packaging standards and carrier performance; streamline return process.",
                "urgency": "Medium",
            }
        )

    # 4. Price complaints
    price_reviews = df[df["review_text"].str.lower().str.contains("price|expensive|waste of money|worth")]
    if not price_reviews.empty:
        recs.append(
            {
                "area": "Pricing Perception",
                "finding": f"{len(price_reviews)} reviews reference price or value concerns.",
                "action": "Review pricing strategy; consider value-add messaging or mid-tier options.",
                "urgency": "Low" if len(price_reviews) < 8 else "Medium",
            }
        )

    # 5. Trending topic that's mostly negative
    if topic_df is not None and not topic_df.empty:
        all_processed = [preprocess(t) for t in df["review_text"]]
        vec = TfidfVectorizer(max_df=0.8, min_df=2, max_features=1000)
        tfidf_vec = vec.fit_transform(all_processed)
        feat = vec.get_feature_names_out()
        nmf_temp = NMF(n_components=min(5, tfidf_vec.shape[1] - 1), random_state=42, init="nndsvdar")
        nmf_temp.fit(tfidf_vec)
        # find the topic with the most negative keywords
        neg_keywords = set("terrible horrible awful broken defect rust overheat leak damage crack".split())
        topic_neg_scores = []
        for comp in nmf_temp.components_:
            top_idx = comp.argsort()[:-11:-1]
            top_words = set(feat[i] for i in top_idx)
            topic_neg_scores.append(len(top_words & neg_keywords))
        worst_topic_idx = int(np.argmax(topic_neg_scores))
        if topic_neg_scores[worst_topic_idx] >= 2:
            top_idx = nmf_temp.components_[worst_topic_idx].argsort()[:-6:-1]
            worst_keywords = [feat[i] for i in top_idx]
            recs.append(
                {
                    "area": "Emerging Complaint Cluster",
                    "finding": f"Topic dominated by negative keywords: {', '.join(worst_keywords)}.",
                    "action": "Investigate this cluster via manual review; prioritize fix in next sprint.",
                    "urgency": "High",
                }
            )

    return recs


def run_full_pipeline(csv_path: str = "data/reviews.csv") -> dict[str, Any]:
    """End-to-end: load → sentiment → topics → trends → recommendations."""
    df = pd.read_csv(csv_path)

    # Sentiment
    sent_df = compute_sentiment(df["review_text"].tolist())
    df = pd.concat([df, sent_df], axis=1)

    # Topics
    topic_df, vectorizer, nmf_model = extract_topics(
        df["review_text"].tolist(), n_topics=5, n_top_words=8
    )
    df["topic_id"] = assign_dominant_topics(
        df["review_text"].tolist(), vectorizer, nmf_model
    )

    # Trends
    trends = detect_trends(df)

    # Recommendations
    recommendations = generate_recommendations(df, topic_df)

    return {
        "data": df,
        "topics": topic_df,
        "trends": trends,
        "recommendations": recommendations,
        "summary": {
            "total_reviews": len(df),
            "avg_rating": round(float(df["rating"].mean()), 2),
            "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
            "category_distribution": df["product_category"].value_counts().to_dict(),
            "topics_found": len(topic_df),
        },
    }
