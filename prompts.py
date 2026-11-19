"""All prompt templates for the review analyzer.

Techniques used:
- Persona: SYSTEM_ANALYST
- Few-shot: build_review_prompt (classification examples)
- Zero-shot: build_cluster_label_prompt
- Chain-of-thought: build_report_prompt (reasoning inside <reasoning> tags)
"""
import json

SYSTEM_ANALYST = (
    "You are a senior customer-experience analyst for an online store. "
    "You are precise and evidence-based. You never invent facts that are not "
    "supported by the reviews you are given."
)

# Few-shot examples for single-review classification.
FEW_SHOT_EXAMPLES = [
    ("Fast delivery and the jacket fits perfectly!",
     {"sentiment": "positive", "topics": ["delivery", "fit"]}),
    ("The battery died after two days and support never answered.",
     {"sentiment": "negative", "topics": ["battery quality", "customer support"]}),
    ("Box arrived on time. The product is fine, nothing special.",
     {"sentiment": "neutral", "topics": ["delivery", "product quality"]}),
]


def build_review_prompt(review: str) -> str:
    """Few-shot prompt that asks for strict JSON output."""
    shots = "\n\n".join(
        f'Review: "{text}"\nOutput: {json.dumps(output)}'
        for text, output in FEW_SHOT_EXAMPLES
    )
    return (
        "Classify the customer review below. Respond ONLY with a JSON object with "
        'the keys "sentiment" (one of: positive, neutral, negative) and "topics" '
        "(1 to 3 short lowercase topic labels).\n\n"
        f"Examples:\n\n{shots}\n\n"
        f'Review: "{review}"\nOutput:'
    )


def build_cluster_label_prompt(samples: list[str]) -> str:
    """Zero-shot prompt that names the theme of a cluster."""
    bullets = "\n".join(f"- {s}" for s in samples)
    return (
        "Below are customer reviews that share a common theme.\n"
        "Give a 2 to 4 word label describing this theme. "
        "Return only the label, nothing else.\n\n" + bullets
    )


def build_report_prompt(digest: str, language: str) -> str:
    """Chain-of-thought prompt for the final executive report."""
    return f"""Analyze the aggregated review statistics below.

Think step by step inside <reasoning> tags:
1. Which sentiment dominates, and how strongly?
2. Which topics appear most often, and do they agree with the clusters?
3. Which problems are severe and frequent, and which are minor?
4. Are there any contradictions or weak signals?

Then write the final report inside <report> tags, in {language}, with these sections:
- Overall sentiment (one short paragraph)
- Top 3 problems, each with one concrete suggested action (bullets)
- Top 3 strengths (bullets)

Use only information present in the statistics below.

Statistics:
{digest}
"""