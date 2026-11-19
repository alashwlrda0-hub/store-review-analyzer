# Store Review Insight Analyzer

A Streamlit application that analyzes online store reviews using a generative AI model.

## Features
- Per-review sentiment and topic classification (few-shot prompting, JSON output)
- Semantic clustering of reviews (sentence embeddings + K-means)
- Zero-shot labeling of each cluster
- Executive report with chain-of-thought reasoning and a customer-experience analyst persona
- Adjustable parameters: model, temperature, top-p, max tokens, cluster count, report language
- Error handling for empty input, too few reviews, invalid API key, rate limits, timeouts, and malformed output
- CSV upload and export

## Setup
1. `conda create --name genai python=3.10` (or add `--solver=classic` if this fails)
2. `conda activate genai`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.
5. Run the app: `streamlit run app.py`

The first run downloads the `all-MiniLM-L6-v2` embedding model, which requires an internet connection.

## Running tests
`pip install pytest` then `pytest test_analyzer.py`

## Prompt techniques
| Technique | Where | Purpose |
|---|---|---|
| Few-shot | `build_review_prompt` | Consistent JSON classification |
| Zero-shot | `build_cluster_label_prompt` | Naming clusters without examples |
| Chain-of-thought | `build_report_prompt` | Reasoning before the report |
| Persona | `SYSTEM_ANALYST` | Analyst tone and grounding |

## Design decisions
- **Temperature 0.2 by default:** classification should be stable; the report is also grounded in statistics, so low randomness is preferred.
- **Clustering before summarizing:** the model sees aggregated statistics and cluster labels, not hundreds of raw reviews, which keeps prompts short and reduces cost.
- **Retries only for transient errors:** rate limits, timeouts, and connection errors are retried with exponential backoff. Authentication errors are not retried.
- **Robust parsing:** malformed JSON falls back to "unknown" instead of crashing the app.