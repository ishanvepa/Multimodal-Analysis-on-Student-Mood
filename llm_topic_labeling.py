"""
Generate human-readable topic names and descriptions from BERTopic outputs
using gpt-oss served via Ollama.

Reads:  bertopic_outputs_{SCHOOL}/topic_info.csv
Writes: bertopic_outputs_{SCHOOL}/LLM_topics.csv

Each row: topic, count, keywords, llm_name, llm_description
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd
import ollama


OLLAMA_MODEL = "gpt-oss"
NUM_KEYWORDS = 10
NUM_DOCS = 10                        # max docs included in the prompt
NUM_EXTRA_DOCS = NUM_DOCS - 3        # random samples beyond the 3 Representative_Docs


SYSTEM_PROMPT = """You are labeling topics discovered by a topic model over Reddit posts from a university subreddit.

For each topic you will receive:
- A ranked list of keywords (unigrams and bigrams) that characterize the topic.
- Several example documents from the topic (some are centroid-closest, others are random samples — use them together to infer the shared theme).

Return a JSON object with exactly two fields:
- "name": a short, human-readable topic name (2-5 words, Title Case, no trailing punctuation).
- "description": a single sentence (max ~25 words) describing what this topic is about in plain English.

IMPORTANT: label the BROADER theme that unifies the examples, not the specific subject of any single post. 
If one document is about making mac and cheese but the keywords and other documents suggest a dining / food-discussion cluster, the topic is "Dining & Food", not "Mac and Cheese Recipes". Prefer general categories a university newspaper would use (academics, housing, events, safety, social life, dining, athletics, etc.) over literal content summaries of individual posts.

Do not include any prose outside the JSON object. Do not wrap the JSON in markdown fences."""


USER_TEMPLATE = """Topic keywords (most → least representative):
{keywords}

Example documents from this topic:
{docs}

Return JSON with "name" and "description" describing the BROADER theme."""


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
        for i, d in enumerate(docs[:NUM_DOCS])
    ) or "(no representative documents available)"
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
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2},
        format="json",
    )
    content = resp["message"]["content"]
    parsed = extract_json(content)
    return parsed.get("name", "").strip(), parsed.get("description", "").strip()


def sample_extra_docs(doc_df, topic_id, k, seed=42):
    """Random sample of `k` additional docs from the topic to broaden LLM context."""
    if doc_df is None or k <= 0:
        return []
    pool = doc_df.loc[doc_df["topic"] == topic_id, "text"].dropna()
    if pool.empty:
        return []
    n = min(k, len(pool))
    return pool.sample(n=n, random_state=seed).tolist()


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
    print(f"Loaded {len(df)} topics from {info_path}")

    records = []
    for _, row in df.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue  # noise cluster — not worth labeling

        keywords = parse_list_cell(row["Representation"])
        repr_docs = parse_list_cell(row["Representative_Docs"])
        extra_docs = sample_extra_docs(doc_df, topic_id, NUM_EXTRA_DOCS)
        docs = repr_docs + extra_docs  # centroid-closest first, then random

        print(f"[topic {topic_id}] labeling… keywords={keywords[:5]} "
              f"({len(repr_docs)} repr + {len(extra_docs)} sampled)")
        try:
            name, desc = label_topic(keywords, docs)
        except Exception as e:
            print(f"  ! failed: {e}")
            name, desc = "", ""

        records.append({
            "topic": topic_id,
            "count": int(row.get("Count", 0)),
            "keywords": ", ".join(keywords[:NUM_KEYWORDS]),
            "llm_name": name,
            "llm_description": desc,
        })
        print(f"  → {name} :: {desc}")

    out_path = out_dir / "LLM_topics.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved {len(records)} labeled topics → {out_path}")


if __name__ == "__main__":
    main()
