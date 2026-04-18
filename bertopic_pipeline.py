import html
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
import sys

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


docs_all = []
meta = []

for d in data:
    raw = f"{d.get('Title','')} {d.get('Text','')}"
    text = clean_text(raw)

    if len(text.split()) > 5:
        docs_all.append(text)

        ts = d.get("Timestamp")
        meta.append({
            "time_period": ts[:7] if isinstance(ts, str) and len(ts) >= 7 else None,
            "emotion": d.get("Emotion"),
            "school": d.get("School"),
            "author": d.get("Author")
        })

print(f"Clean docs: {len(docs_all)}")


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


# ================================
# TOPIC NAMING (CLEAN VERSION)
# ================================
def make_topic_name(keywords):
    words = [w for w, _ in keywords[:5]]

    if any("housing" in w or "roommate" in w for w in words):
        return "Housing & Roommates"
    if any("grade" in w or "gpa" in w for w in words):
        return "Grades & GPA"
    if any("club" in w or "friend" in w for w in words):
        return "Clubs & Social Life"
    if any("parking" in w for w in words):
        return "Parking & Transportation"
    if any("admission" in w or "transfer" in w for w in words):
        return "Admissions & Transfers"
    if any("graduation" in w or "commencement" in w for w in words):
        return "Graduation"
    if any("ticket" in w or "event" in w for w in words):
        return "Events"

    return "Other Student Discussions"


topic_labels = {}

for t in topic_model.get_topic_info()["Topic"]:
    if t == -1:
        continue
    topic_labels[t] = make_topic_name(topic_model.get_topic(t))


# APPLY LABELS
topic_model.set_topic_labels(topic_labels)

df["topic_name"] = df["topic"].map(topic_labels)


# ================================
# SAVE FINAL DATASET FOR DASHBOARD
# ================================
df.to_csv(out_dir / "dashboard_data.csv", index=False)

print("Pipeline complete")