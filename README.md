# Customer Feedback Intelligence System

[![Status: Live](https://img.shields.io/badge/status-live-22c55e.svg)](https://github.com/agenticph-labs/p4-customer-feedback-intel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An NLP-powered system that analyzes customer feedback data — extracts sentiment, discovers topics, detects trends, and generates actionable recommendations.

## 📊 Dashboard Preview

| Sentiment Overview | Topic Discovery | Trend Analysis | Recommendations |
|---|---|---|---|
| Pie chart + KPI cards | TF-IDF + NMF topic clusters | Monthly sentiment/rating trends | Urgency-ranked actions |

## 🧠 Methodology

### 1. Sentiment Analysis — [VADER](https://github.com/cjhutto/vaderSentiment)
- **Valence Aware Dictionary and sEntiment Reasoner** — a rule-based lexicon optimized for social-media-style text.
- Each review gets a *compound* score (`-1.0` to `+1.0`), mapped to:
  - **Positive** (≥ 0.05)
  - **Neutral** (> -0.05, < 0.05)
  - **Negative** (≤ -0.05)
- VADER handles intensifiers, negations, and emoticons out of the box.

### 2. Topic Extraction — TF-IDF + NMF
- **TF-IDF Vectorization**: converts review text into a weighted term-frequency matrix; `max_df=0.8` and `min_df=2` filter very common and very rare terms.
- **Non-Negative Matrix Factorization (NMF)**: decomposes the TF-IDF matrix into `n` topics (default 5) with additive, interpretable components.
- Each topic is summarized by its top 8 keywords, and every review is assigned its dominant topic.

### 3. Trend Detection — Temporal Aggregation
- Reviews are grouped by calendar month.
- Monthly averages for **rating** and **sentiment volume** are computed to surface shifts over time.
- Combined dual-axis chart (rating line + review-count bars) reveals correlation between volume and satisfaction.

### 4. Action Recommendations — Rule-Based Heuristics
A set of hand-crafted rules inspects the analyzed data to derive business actions:
- **Product Quality Issues**: keywords from low-rated (≤ 2) reviews.
- **Category Sentiment Hotspots**: the category with the most negative reviews.
- **Shipping & Fulfillment**: reviews mentioning shipping, delivery, returns, or damage.
- **Pricing Perception**: reviews citing price/value.
- **Emerging Complaint Cluster**: NMF topic with the highest negative-keyword density.

## 📁 Project Structure

```
p4-customer-feedback-intel/
├── data/
│   └── reviews.csv          # 100 sample customer reviews (Jan–Sep 2025)
├── nlp_pipeline.py          # Core NLP pipeline module
├── dashboard.py             # Streamlit dashboard
├── requirements.txt         # Python dependencies
├── Makefile                 # Convenience targets
└── README.md                # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [pip](https://pip.pypa.io/) or [uv](https://docs.astral.sh/uv/)

### Installation

```bash
# Clone the repository
git clone https://github.com/agenticph-labs/p4-customer-feedback-intel.git
cd p4-customer-feedback-intel

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run dashboard.py
```

Then open the URL shown in your terminal (typically `http://localhost:8501`).

### Run the Pipeline (CLI)

```python
from nlp_pipeline import run_full_pipeline

results = run_full_pipeline("data/reviews.csv")
print(results["summary"])
```

## 📊 Sample Dataset

The bundled dataset (`data/reviews.csv`) contains **100 synthetic customer reviews** across 3 product categories:

| Category | Reviews | Rating Range |
|---|---|---|
| Electronics | 40 | 1–5 |
| Home & Kitchen | 30 | 1–5 |
| Clothing | 30 | 1–5 |

Reviews span **January – September 2025** to support trend detection.

## 🔧 Customization

- **Add your own data**: place a CSV with columns `id`, `date`, `product_category`, `rating`, `review_text` in `data/`.
- **Adjust topic count**: change `n_topics` in the `run_full_pipeline()` call.
- **Add categories**: the pipeline automatically handles any product category labels.

## 📈 Outputs

| Output | Description |
|---|---|
| Sentiment Distribution | Positive / Neutral / Negative percentages |
| Extracted Topics | 5 topic clusters with top keywords |
| Monthly Trends | Rating average + sentiment volume over time |
| Action Recommendations | 3–5 urgency-ranked business actions |
| Filtered Data | Interactive filters by category, sentiment, rating |

## 🛠️ Tech Stack

- **Python** — Core logic and NLP pipeline
- **NLTK (VADER)** — Sentiment analysis
- **scikit-learn** — TF-IDF vectorization + NMF topic modeling
- **Streamlit** — Interactive dashboard
- **Plotly** — Charts and visualizations
- **Pandas** — Data manipulation

## 📄 License

MIT

---

*Portfolio Project 4 — [AgenticPH Labs](https://agenticph-labs.github.io/portfolio)*
