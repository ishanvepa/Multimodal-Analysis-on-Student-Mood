import html
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# ================================
# CONFIG
# ================================

school = sys.argv[1]
if school not in ["UNC", "GATECH"]:
    raise ValueError("Invalid school. Must be 'UNC' or 'GATECH'.")
out_dir = Path(f"bertopic_outputs_{school}")
n_topics = 30
out_dir.mkdir(exist_ok=True)

file_source = f"cleaned_reddit_reviews/cleaned_redditreviews_{school}.json.zip"

# TODO: run it through other people to converge the choice of stopwords.

# Only strip tokens that are genuinely information-less for topic analysis.
# Kept (signals?): atlanta, buzz, jackets, chapel hill, carolina, etc.
'''
sklearn only looks up individual tokens. bigrams stopwords are not supported
if needed, manual filtering in the vectorizer and then again in the topic naming step.
'''
SCHOOL_STOPWORDS = {
    "UNC": {"unc", "runc", "tarheel", "tarheels", "tar", "heel"},
    "GATECH": {"gatech", "rgatech", "tech", "gt", "georgia"},
}
# Reddit mechanics + HTML artifacts + contraction filler (no topic signal)
GENERIC_STOPWORDS = {
    "subreddit", "moderators", "automatically", "bot", "flair", "karma",
    "amp", "x200b", "nbsp",
    "im", "dont", "youre", "youve", "ive", "thats", "wasnt", "didnt",
    # "lol", "tbh", "idk", "imo", "omg", "yeah",
    # "just", "like", "really", "know", "think",
}
CUSTOM_STOPWORDS = list(
    ENGLISH_STOP_WORDS
    | GENERIC_STOPWORDS
    | SCHOOL_STOPWORDS.get(school, set())
)

# Posts matching these patterns are bot/automod messages — remove entirely
BOT_PATTERNS = [
    re.compile(r"hello /?u/\S+,?\s+(welcome to|it looks like)", re.I),
    re.compile(r"your (comment|submission|post) has been automatically removed", re.I),
    re.compile(r"i am a bot, and this action was performed automatically", re.I),
    re.compile(r"it appears you may have lost or found something", re.I),
    re.compile(r"please \[choose a user flair\]", re.I),
    re.compile(r"this is a mentally ill spammer", re.I),
]


# ================================
# LOAD DATA
# ================================
with zipfile.ZipFile(file_source, "r") as z:
    json_files = [f for f in z.namelist() if f.endswith(".json")]
    if not json_files:
        raise ValueError("No JSON file found in zip.")

    with z.open(json_files[0]) as f:
        data = json.load(f)

print(f"Loaded {len(data)} records")


# ================================
# CLEAN TEXT
# ================================
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    if text in ["[deleted]", "[removed]"]:
        return ""

    # Drop bot / automod posts before any further cleaning
    if any(p.search(text) for p in BOT_PATTERNS):
        return ""

    # Decode HTML entities (&amp;, &gt;, &#x200b;) and strip zero-width space
    text = html.unescape(text)
    text = re.sub(r"\u200b", "", text)

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"/?u/\S+|/?r/\S+", "", text)   # /u/user, /r/sub
    text = re.sub(r">", " ", text)
    text = re.sub(r"\*\*|\*", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ================================
# DEDUP (MinHash + LSH, near-duplicate detection)
# ================================
# Reddit dumps contain reposts / crossposts where the same body differs
# only by a 1-2 sentence prefix. Jaccard ~0.85 on 5-word shingles catches
# those without removing genuine quote-and-reply posts (Jaccard ~0.7).
SHINGLE_K = 5
LSH_THRESHOLD = 0.9
LSH_NUM_PERM = 128


def minhash_of(text):
    tokens = text.split()
    if len(tokens) < SHINGLE_K:
        shingles = {text}
    else:
        shingles = {
            " ".join(tokens[i:i + SHINGLE_K])
            for i in range(len(tokens) - SHINGLE_K + 1)
        }
    mh = MinHash(num_perm=LSH_NUM_PERM)
    for s in shingles:
        mh.update(s.encode("utf-8"))
    return mh


# ---- Identify posts vs comments ----
# The scraper propagates a post's Title to every comment under it, so
# raw Title+Text concatenation would force-feed the post title into every
# comment and bias topic modeling toward post titles. Heuristic: within
# records sharing a Title, the earliest by Timestamp is the post; the rest
# are comments. Posts use Title+Text; comments use Text only.
post_by_title = {}   # title -> (earliest_ts, record_idx)
for i, d in enumerate(data):
    title = d.get("Title") or ""
    if not title:
        continue
    ts = d.get("Timestamp") or ""
    cur = post_by_title.get(title)
    if cur is None or (ts and ts < cur[0]):
        post_by_title[title] = (ts, i)

post_idx_set = {idx for _, idx in post_by_title.values()}


lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=LSH_NUM_PERM)
docs_all = []
meta = []
n_exact_dup = 0
n_near_dup = 0
n_posts = 0
n_comments = 0
seen_exact = set()
drop_samples = []   # first ~20 (kept_text, dropped_text) pairs for eyeballing

for i, d in enumerate(data):
    title = d.get("Title") or ""
    text_body = d.get("Text") or ""

    if title and i in post_idx_set:
        raw = f"{title} {text_body}".strip()
        is_post = True
    else:
        # comment (shares title with an earlier record) or orphan (no title)
        raw = text_body
        is_post = False

    text = clean_text(raw)

    if len(text.split()) <= 5:
        continue

    if text in seen_exact:
        n_exact_dup += 1
        continue

    mh = minhash_of(text)
    matches = lsh.query(mh)
    if matches:
        n_near_dup += 1
        if len(drop_samples) < 20:
            kept_idx = int(matches[0].split("_", 1)[1])
            drop_samples.append((docs_all[kept_idx], text))
        continue

    key = f"doc_{len(docs_all)}"
    lsh.insert(key, mh)
    seen_exact.add(text)

    if is_post:
        n_posts += 1
    else:
        n_comments += 1

    ts = d.get("Timestamp")
    meta.append({
        "time_period": ts[:7] if isinstance(ts, str) and len(ts) >= 7 else None,
        "emotion": d.get("Emotion"),
        "school": d.get("School"),
        "author": d.get("Author"),
        "is_post": is_post,
    })
    docs_all.append(text)

print(f"Clean docs: {len(docs_all)} "
      f"({n_posts} posts + {n_comments} comments; "
      f"dropped {n_exact_dup} exact + {n_near_dup} near duplicates)")

if drop_samples:
    print("\n--- sample near-duplicate pairs (first 20) ---")
    for i, (kept, dropped) in enumerate(drop_samples, 1):
        print(f"\n[pair {i}]")
        print(f"  KEPT:    {kept[:240]}…")
        print(f"  DROPPED: {dropped[:240]}…")
    print("--- end samples ---\n")


# ================================
# MODEL
# ================================
vectorizer = CountVectorizer(stop_words=CUSTOM_STOPWORDS, ngram_range=(1, 2))
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic(
    # embedding_model=embedding_model,
    vectorizer_model=vectorizer,
    min_topic_size=15,
    nr_topics=n_topics,
    verbose=True,
)


topics, probs = topic_model.fit_transform(docs_all)

topics = topic_model.reduce_outliers(docs_all, topics, probabilities=probs)

# Sync reduced-outlier assignments back into the model so get_topic_info()
# and visualizations reflect the post-reduction state.
topic_model.update_topics(docs_all, topics=topics, vectorizer_model=vectorizer)


# ================================
# DATAFRAME
# ================================
df = pd.DataFrame({
    "text": docs_all,
    "topic": topics
})

meta_df = pd.DataFrame(meta)

df = pd.concat([df, meta_df], axis=1)


# ================================
# SAVE BASE OUTPUT
# ================================
df.to_csv(out_dir / "doc_topics.csv", index=False)
topic_model.get_topic_info().to_csv(out_dir / "topic_info.csv", index=False)

# Interactive HTML visualization (if running locally)
try:
    fig = topic_model.visualize_topics()
    fig.write_html(f"{out_dir}/topics_viz.html")
except Exception as e:
    print(f"\nSkipped interactive viz: {e}")

# Bar chart of top words per topic
try:
    fig_bar = topic_model.visualize_barchart(top_n_topics=n_topics)
    fig_bar.write_html(f"{out_dir}/barchart.html")
    print(f"Saved bar chart → {out_dir}/barchart.html")
except Exception as e:
    print(f"Skipped bar chart viz: {e}")


# Topic names are produced separately by llm_topic_labeling.py.
# To get a doc-level table with human-readable topic names, merge
# doc_topics.csv with LLM_topics.csv on the `topic` column.

print("Pipeline complete")