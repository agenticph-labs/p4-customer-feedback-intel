.PHONY: install run test clean

install:
	pip install -r requirements.txt

run:
	streamlit run dashboard.py

test:
	python -c "from nlp_pipeline import run_full_pipeline; r = run_full_pipeline('data/reviews.csv'); print('Pipeline OK'); print(f'Sentiment: {r[\"summary\"][\"sentiment_distribution\"]}'); print(f'Topics: {len(r[\"topics\"])}'); print(f'Recommendations: {len(r[\"recommendations\"])}')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name '*.pyc' -delete
