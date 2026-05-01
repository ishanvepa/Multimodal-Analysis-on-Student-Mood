"""
Posts-only BERTopic pipeline.

Fits BERTopic on posts only, then propagates each post's topic down to its
comments via a thread-key heuristic (Title + time-gap split). Orphan comments
(no parent post in the data) are dropped. The doc_topics.csv output therefore
covers (posts + attached comments) and supports the same downstream emotion
analysis with a sharper topic backbone.

Why this exists: the original pipeline concatenates Title + Text for *every*
record, so comments inherit their parent post's title and cluster by thread
rather than by their own content. That produces a 40k-doc catch-all topic
and incoherent topic labels (e.g. a "Career & Employment" topic glued by the
word "letter" — cover letters, recommendation letters, letter grades).
"""
import html
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# ================================
# CONFIG
# ================================
school = sys.argv[1]
if school not in ["UNC", "GATECH"]:
    raise ValueError("Invalid school. Must be 'UNC' or 'GATECH'.")
out_dir = Path(f"bertopic_outputs_posts_{school}")
out_dir.mkdir(exist_ok=True)
n_topics = 30
THREAD_GAP_DAYS = 14   # records sharing a Title with a >14d gap = different threads
MIN_WORDS = 6           # drop posts/comments with <= 5 words

file_source = f"processed_reddit_reviews/processed_redditreviews_{school}.json"

SCHOOL_STOPWORDS = {
    "UNC": {"unc", "runc", "tarheel", "tarheels", "tar", "heel"},
    "GATECH": {"gatech", "rgatech", "tech", "gt", "georgia"},
}
GENERIC_STOPWORDS = {
    "subreddit", "moderators", "automatically", "bot", "flair", "karma",
    "amp", "x200b", "nbsp",
    "im", "dont", "youre", "youve", "ive", "thats", "wasnt", "didnt",
}
CUSTOM_STOPWORDS = list(
    ENGLISH_STOP_WORDS | GENERIC_STOPWORDS | SCHOOL_STOPWORDS.get(school, set())
)

BOT_PATTERNS = [
    re.compile(r"hello /?u/\S+,?\s+(welcome to|it looks like)", re.I),
    re.compile(r"your (comment|submission|post) has been automatically removed", re.I),
    re.compile(r"i am a bot, and this action was performed automatically", re.I),
    re.compile(r"it appears you may have lost or found something", re.I),
    re.compile(r"please \[choose a user flair\]", re.I),
    re.compile(r"this is a mentally ill spammer", re.I),
]


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    if text in ["[deleted]", "[removed]"]:
        return ""
    if any(p.search(text) for p in BOT_PATTERNS):
        return ""
    text = html.unescape(text)
    text = re.sub(r"​", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"/?u/\S+|/?r/\S+", "", text)
    text = re.sub(r">", " ", text)
    text = re.sub(r"\*\*|\*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


# ================================
# LOAD
# ================================
data = json.load(open(file_source))
print(f"Loaded {len(data):,} records from {file_source}")


# ================================
# THREAD-KEYING
# Group records by Title; within each title, walk in timestamp order and
# open a new thread whenever the gap from the previous record exceeds
# THREAD_GAP_DAYS. Earliest record in each bucket is the post; the rest
# are its comments.
# ================================
no_title_idxs = [i for i, d in enumerate(data) if not (d.get("Title") or "").strip()]
n_orphan = len(no_title_idxs)

by_title = defaultdict(list)
for i, d in enumerate(data):
    t = (d.get("Title") or "").strip()
    if not t:
        continue
    by_title[t].append((i, parse_ts(d.get("Timestamp"))))

threads = []  # each: {"post_idx": int, "comments": [int], "title": str}
for title, items in by_title.items():
    items.sort(key=lambda x: (x[1] is None, x[1]))  # None timestamps go last
    cur = [items[0]]
    for i, ts in items[1:]:
        prev_ts = cur[-1][1]
        if prev_ts and ts and (ts - prev_ts) > timedelta(days=THREAD_GAP_DAYS):
            threads.append({
                "post_idx": cur[0][0],
                "comments": [x[0] for x in cur[1:]],
                "title": title,
            })
            cur = [(i, ts)]
        else:
            cur.append((i, ts))
    threads.append({
        "post_idx": cur[0][0],
        "comments": [x[0] for x in cur[1:]],
        "title": title,
    })

n_attached = sum(len(t["comments"]) for t in threads)
print(f"\nThread-keying:")
print(f"  unique titles:                {len(by_title):,}")
print(f"  threads after gap-split (>{THREAD_GAP_DAYS}d): {len(threads):,}")
print(f"  posts (thread anchors):       {len(threads):,}")
print(f"  comments attached to a post:  {n_attached:,}")
print(f"  orphan comments (no Title):   {n_orphan:,} ({100*n_orphan/len(data):.1f}%)  [DROPPED]")


# ================================
# BUILD POST DOCS (title + body) FOR TOPIC MODELING
# Comments are NOT fed into the topic model; they inherit their post's topic.
# ================================
post_docs_raw = []
post_meta = []
for tk in threads:
    p_idx = tk["post_idx"]
    d = data[p_idx]
    raw = ((d.get("Title") or "") + " " + (d.get("Text") or "")).strip()
    text = clean_text(raw)
    post_docs_raw.append(text)
    post_meta.append({
        "Unique_ID": d.get("Unique_ID"),
        "post_idx_in_data": p_idx,
        "thread_size": 1 + len(tk["comments"]),
    })

# Drop too-short posts AND exact-duplicate posts (rare at this scale, but cheap)
seen = set()
keep_ix = []
for i, text in enumerate(post_docs_raw):
    if len(text.split()) < MIN_WORDS:
        continue
    if text in seen:
        continue
    seen.add(text)
    keep_ix.append(i)

post_docs = [post_docs_raw[i] for i in keep_ix]
post_meta = [post_meta[i] for i in keep_ix]
post_threads = [threads[i] for i in keep_ix]
print(f"  posts after short/exact-dup filter: {len(post_docs):,}")


# ================================
# MODEL
# ================================
vectorizer = CountVectorizer(stop_words=CUSTOM_STOPWORDS, ngram_range=(1, 2))
topic_model = BERTopic(
    vectorizer_model=vectorizer,
    min_topic_size=15,
    nr_topics=n_topics,
    verbose=True,
)
topics, probs = topic_model.fit_transform(post_docs)
topics = topic_model.reduce_outliers(post_docs, topics, probabilities=probs)
topic_model.update_topics(post_docs, topics=topics, vectorizer_model=vectorizer)


# ================================
# DOC-LEVEL OUTPUT: posts + their attached (cleanable) comments
# Each comment inherits its post's topic. Comments with too-short cleaned
# text are still dropped so downstream emotion lift isn't polluted by noise.
# ================================
out_rows = []
for meta, topic, tk in zip(post_meta, topics, post_threads):
    p = data[meta["post_idx_in_data"]]
    out_rows.append({
        "Unique_ID": p.get("Unique_ID"),
        "topic": int(topic),
        "school": p.get("School"),
        "is_post": True,
    })
    for c_idx in tk["comments"]:
        c = data[c_idx]
        c_text = clean_text(c.get("Text") or "")
        if len(c_text.split()) < MIN_WORDS:
            continue
        out_rows.append({
            "Unique_ID": c.get("Unique_ID"),
            "topic": int(topic),
            "school": c.get("School"),
            "is_post": False,
        })

doc_df = pd.DataFrame(out_rows)
print(f"\nWrote {len(doc_df):,} doc-topic rows "
      f"({(doc_df['is_post']).sum():,} posts + "
      f"{(~doc_df['is_post']).sum():,} comments)")

doc_df.to_csv(out_dir / "doc_topics.csv", index=False)
topic_model.get_topic_info().to_csv(out_dir / "topic_info.csv", index=False)

try:
    fig = topic_model.visualize_topics()
    fig.write_html(f"{out_dir}/topics_viz.html")
except Exception as e:
    print(f"Skipped interactive viz: {e}")
try:
    fig_bar = topic_model.visualize_barchart(top_n_topics=n_topics)
    fig_bar.write_html(f"{out_dir}/barchart.html")
except Exception as e:
    print(f"Skipped bar chart viz: {e}")

print("Pipeline complete")
