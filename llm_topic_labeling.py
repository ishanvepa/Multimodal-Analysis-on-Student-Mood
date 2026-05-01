"""
Generate human-readable topic names and descriptions from BERTopic outputs
using a model served via vLLM's OpenAI-compatible HTTP server.

Reads:  bertopic_outputs_{SCHOOL}/topic_info.csv
Writes: bertopic_outputs_{SCHOOL}/LLM_topics.csv

Each row: topic, count, keywords, llm_name, llm_description, coherence
"""

import argparse
import ast
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI

from bertopic_pipeline import clean_text


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = ""
VLLM_MODEL = "openai/gpt-oss-20b"

client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

NUM_KEYWORDS = 10
NUM_REP_DOCS = 3
NUM_RANDOM_DOCS = 15
DOC_SHUFFLE_SEED = 7


SYSTEM_PROMPT = """You are labeling topics discovered by a topic model over Reddit posts from a university subreddit.

For each topic you will receive:
- A ranked list of keywords (unigrams and bigrams) that characterize the topic.
- A set of example documents drawn from the topic.

Return a JSON object with exactly three fields:
- "name": a short, human-readable topic name (2-5 words, Title Case, no trailing punctuation).
- "description": a single sentence (max ~25 words) describing what this topic is about in plain English.
- "coherence": one of "high", "mixed", or "low". Use "high" when most documents share an obvious theme; "mixed" when there are 2-3 distinct sub-themes; "low" when the documents look like a residual catch-all with no unifying theme.

IMPORTANT:
- Weigh the keywords and ALL documents together. No document is more authoritative than the others — read across them and judge what fraction actually fits a candidate theme. If only a minority of documents support the theme suggested by the top keywords, the cluster is likely mixed or low coherence; do not invent a unifying label.
- Label the BROADER theme that unifies the examples, not the specific subject of any single post. If one document is about making mac and cheese but other documents suggest a dining cluster, the topic is "Dining & Food", not "Mac and Cheese Recipes". Prefer general categories a university newspaper would use (academics, housing, events, safety, social life, dining, athletics, etc.) over literal content summaries of individual posts.
- For "mixed" or "low" coherence topics, the description should briefly list the sub-themes you see (e.g. "Heterogeneous cluster of campus discourse: politics, religion on campus, admin meta-complaints"). Do not force a single theme.

Do not include any prose outside the JSON object. Do not wrap the JSON in markdown fences."""


USER_TEMPLATE = """Topic keywords (most → least representative):
{keywords}

Example documents from this topic:
{docs}

Return JSON with "name", "description", and "coherence"."""


def parse_list_cell(cell):
    """topic_info.csv stores Python-list-literal strings in Representation / Representative_Docs."""
    if pd.isna(cell):
        return []
    try:
        val = ast.literal_eval(cell)
        return val if isinstance(val, list) else [str(val)]
    except (ValueError, SyntaxError):
        return [str(cell)]


def truncate(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def build_user_prompt(keywords, docs):
    kw_str = ", ".join(keywords[:NUM_KEYWORDS])
    doc_str = "\n\n".join(
        f"[{i+1}] {d}"
        for i, d in enumerate(docs)
    ) or "(no documents available)"
    return USER_TEMPLATE.format(keywords=kw_str, docs=doc_str)


def extract_json(text):
    """gpt-oss usually returns clean JSON; be defensive in case of stray prose or fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return json.loads(text[start:end + 1])


def label_topic(keywords, docs):
    user_prompt = build_user_prompt(keywords, docs)
    resp = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    parsed = extract_json(content)
    coherence = parsed.get("coherence", "").strip().lower()
    if coherence not in {"high", "mixed", "low"}:
        coherence = ""
    return (
        parsed.get("name", "").strip(),
        parsed.get("description", "").strip(),
        coherence,
    )


def sample_extra_docs(doc_df, topic_id, k, seed=42):
    """Random sample of `k` additional docs from the topic to broaden LLM context."""
    if doc_df is None or k <= 0:
        return []
    pool = doc_df.loc[doc_df["topic"] == topic_id, "text"].dropna()
    if pool.empty:
        return []
    n = min(k, len(pool))
    return pool.sample(n=n, random_state=seed).tolist()


def load_text_by_id(school):
    """Build Unique_ID -> text map from the processed source JSON.

    doc_topics.csv only carries Unique_ID now, so we reattach the body
    (Title + Text for posts, Text alone for comments) here.
    """
    src = Path(f"processed_reddit_reviews/processed_redditreviews_{school}.json")
    if not src.exists():
        return {}
    with open(src, "r") as f:
        data = json.load(f)
    out = {}
    for d in data:
        uid = d.get("Unique_ID")
        if uid is None:
            continue
        title = d.get("Title") or ""
        body = d.get("Text") or ""
        raw = f"{title} {body}".strip() if title else body
        cleaned_text = clean_text(raw)
        # drop empty texts
        if not cleaned_text:
            continue
        out[uid] = cleaned_text
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--school", choices=["UNC", "GATECH"], required=True)
    args = parser.parse_args()

    out_dir = Path(f"bertopic_outputs_{args.school}")
    info_path = out_dir / "topic_info.csv"
    docs_path = out_dir / "doc_topics.csv"
    if not info_path.exists():
        sys.exit(f"Missing {info_path}. Run bertopic_pipeline.py first.")

    df = pd.read_csv(info_path)
    doc_df = pd.read_csv(docs_path) if docs_path.exists() else None
    if doc_df is None:
        print(f"(doc_topics.csv not found — using only Representative_Docs)")
    else:
        text_by_id = load_text_by_id(args.school)
        if text_by_id:
            doc_df["text"] = doc_df["Unique_ID"].map(text_by_id)
        else:
            print(f"(source JSON for {args.school} not found — extra-doc sampling disabled)")
            doc_df = None
    print(f"Loaded {len(df)} topics from {info_path}")

    records = []
    for _, row in df.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue  # noise cluster — not worth labeling

        keywords = parse_list_cell(row["Representation"])
        repr_docs = parse_list_cell(row["Representative_Docs"])[:NUM_REP_DOCS]
        extra_docs = sample_extra_docs(doc_df, topic_id, NUM_RANDOM_DOCS)
        docs = repr_docs + extra_docs
        random.Random(DOC_SHUFFLE_SEED + topic_id).shuffle(docs)

        print(f"[topic {topic_id}] labeling… keywords={keywords[:5]} "
              f"({len(repr_docs)} repr + {len(extra_docs)} sampled, shuffled)")
        try:
            name, desc, coherence = label_topic(keywords, docs)
        except Exception as e:
            print(f"  ! failed: {e}")
            name, desc, coherence = "", "", ""

        records.append({
            "topic": topic_id,
            "count": int(row.get("Count", 0)),
            "keywords": ", ".join(keywords[:NUM_KEYWORDS]),
            "llm_name": name,
            "llm_description": desc,
            "coherence": coherence,
        })
        print(f"  → [{coherence or '?'}] {name} :: {desc}")

    out_path = out_dir / "LLM_topics.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved {len(records)} labeled topics → {out_path}")


if __name__ == "__main__":
    main()
