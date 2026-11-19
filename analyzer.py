"""Analysis pipeline: LLM calls, clustering, parsing, and error handling."""
import json
import re
import time
from collections import Counter
from dataclasses import dataclass

from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from prompts import (
    SYSTEM_ANALYST,
    build_cluster_label_prompt,
    build_report_prompt,
    build_review_prompt,
)

RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
MIN_REVIEWS = 3
MAX_REVIEW_CHARS = 1000
MAX_REVIEWS = 200


class AnalyzerError(Exception):
    """An error with a message that is safe to show to the user."""


@dataclass
class Settings:
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1500
    clusters: int = 4
    language: str = "English"
    max_retries: int = 3


# ---------- Input cleaning ----------

def clean_reviews(raw_reviews) -> list[str]:
    """Strip whitespace, drop empty and duplicate reviews, and cap length and count."""
    seen, cleaned = set(), []
    for review in raw_reviews:
        text = re.sub(r"\s+", " ", str(review)).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        cleaned.append(text[:MAX_REVIEW_CHARS])
        if len(cleaned) >= MAX_REVIEWS:
            break
    return cleaned


# ---------- Output parsing ----------

def parse_classification(raw: str) -> dict:
    """Extract the JSON object from the model output. Falls back safely if malformed."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    sentiment = str(data.get("sentiment", "unknown")).strip().lower()
    if sentiment not in {"positive", "neutral", "negative"}:
        sentiment = "unknown"
    topics = [str(t).strip().lower() for t in data.get("topics", []) if str(t).strip()]
    return {"sentiment": sentiment, "topics": topics[:3]}


def extract_report(raw: str) -> str:
    """Return the text inside <report> tags, or the full output if the tags are missing."""
    match = re.search(r"<report>(.*?)</report>", raw, re.DOTALL)
    return (match.group(1) if match else raw).strip()


# ---------- LLM access ----------

def make_llm(api_key: str, settings: Settings) -> ChatGroq:
    return ChatGroq(
        api_key=api_key,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        model_kwargs={"top_p": settings.top_p},
        timeout=60,
        max_retries=0,  # retries are handled by _call_llm
    )


def _call_llm(llm: ChatGroq, messages: list, max_retries: int) -> str:
    """Call the model with exponential backoff on transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            content = llm.invoke(messages).content
            if not content or not str(content).strip():
                raise AnalyzerError("The model returned an empty response.")
            return str(content)
        except AuthenticationError as e:
            raise AnalyzerError("Invalid Groq API key. Check the sidebar or your .env file.") from e
        except RETRYABLE_ERRORS as e:
            if attempt == max_retries:
                raise AnalyzerError(
                    f"The API is temporarily unavailable or rate-limited ({type(e).__name__}). "
                    "Please wait a moment and try again."
                ) from e
            time.sleep(2 ** attempt)
        except AnalyzerError:
            raise
        except Exception as e:
            raise AnalyzerError(f"Unexpected API error: {e}") from e


# ---------- Clustering ----------

_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def cluster_reviews(reviews: list[str], k: int) -> list[int]:
    k = max(1, min(k, len(reviews)))
    try:
        embeddings = _get_embedder().encode(reviews, normalize_embeddings=True)
    except Exception as e:
        raise AnalyzerError(
            "Could not load the embedding model. An internet connection is needed on the first run."
        ) from e
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embeddings).tolist()


# ---------- Pipeline ----------

def build_digest(results: list[dict], clusters: dict) -> str:
    sentiment = Counter(r["sentiment"] for r in results)
    topics = Counter(t for r in results for t in r["topics"])
    lines = [
        f"Total reviews: {len(results)}",
        f"Sentiment counts: {dict(sentiment)}",
        "Most frequent topics: " + ", ".join(f"{t} ({c})" for t, c in topics.most_common(8)),
        "Thematic clusters:",
    ]
    lines += [f"- '{c['label']}': {c['size']} reviews" for c in clusters.values()]
    return "\n".join(lines)


def analyze(raw_reviews, settings: Settings, api_key: str, progress=None) -> dict:
    """Run the full pipeline. `progress` is an optional callable that takes a float in [0, 1]."""
    if not api_key:
        raise AnalyzerError("Please enter a Groq API key in the sidebar or in your .env file.")

    reviews = clean_reviews(raw_reviews)
    if len(reviews) < MIN_REVIEWS:
        raise AnalyzerError(
            f"Please provide at least {MIN_REVIEWS} unique, non-empty reviews (got {len(reviews)})."
        )

    llm = make_llm(api_key, settings)
    report_progress = progress or (lambda _: None)

    # Step 1: few-shot classification of each review
    results = []
    for i, review in enumerate(reviews):
        raw = _call_llm(
            llm, [HumanMessage(content=build_review_prompt(review))], settings.max_retries
        )
        results.append({"review": review, **parse_classification(raw)})
        report_progress(0.5 * (i + 1) / len(reviews))

    # Step 2: semantic clustering, then zero-shot labeling of each cluster
    labels = cluster_reviews(reviews, settings.clusters)
    clusters = {}
    unique_labels = sorted(set(labels))
    for n, label in enumerate(unique_labels):
        members = [i for i, l in enumerate(labels) if l == label]
        samples = [reviews[i] for i in members][:8]
        name = _call_llm(
            llm,
            [SystemMessage(content=SYSTEM_ANALYST),
             HumanMessage(content=build_cluster_label_prompt(samples))],
            settings.max_retries,
        ).strip().strip('"')
        clusters[int(label)] = {"label": name, "size": len(members), "reviews": members}
        report_progress(0.5 + 0.2 * (n + 1) / len(unique_labels))

    # Step 3: chain-of-thought executive report
    digest = build_digest(results, clusters)
    raw_report = _call_llm(
        llm,
        [SystemMessage(content=SYSTEM_ANALYST),
         HumanMessage(content=build_report_prompt(digest, settings.language))],
        settings.max_retries,
    )
    report_progress(1.0)

    topic_counts = Counter(t for r in results for t in r["topics"])
    return {
        "results": results,
        "clusters": clusters,
        "topic_counts": dict(topic_counts),
        "report": extract_report(raw_report),
    }