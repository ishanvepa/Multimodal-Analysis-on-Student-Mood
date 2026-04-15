import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer


# ================================
# CONFIG
# ================================
school = "GATECH"

out_dir = Path(f"bertopic_outputs_{school}")
out_dir.mkdir(exist_ok=True)

file_source = f"cleaned_reddit_reviews/cleaned_redditreviews_{school}.json.zip"


# ================================
# STEP 0 — LOAD DATA
# ================================
with zipfile.ZipFile(file_source, "r") as z:
    json_files = [f for f in z.namelist() if f.endswith(".json")]

    if not json_files:
        raise ValueError("No JSON file found in zip.")

    json_name = json_files[0]

    with z.open(json_name) as f:
        data = json.load(f)

print(f"Total records loaded: {len(data)}")


# ================================
# STEP 1 — CLEAN TEXT
# ================================
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    if text in ["[deleted]", "[removed]"]:
        return ""

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r">", " ", text)
    text = re.sub(r"\*\*|\*", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


docs_all = []
meta = []

for d in data:
    title = d.get("Title", "")
    text = d.get("Text", "")

    raw = f"{title} {text}".strip()
    combined = clean_text(raw)

    if len(combined.split()) > 5:
        docs_all.append(combined)

        timestamp = d.get("Timestamp", None)
        time_period = None

        if isinstance(timestamp, str) and len(timestamp) >= 7:
            time_period = timestamp[:7]

        meta.append({
            "time_period": time_period,
            "emotion": d.get("Emotion"),
            "school": d.get("School"),
            "author": d.get("Author")
        })

print(f"Total cleaned docs: {len(docs_all)}")


# ================================
# STEP 2 — VECTORIZE + MODEL
# ================================
custom_stopwords = {"gatech", "gt", "georgia", "tech"}

vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer,
    min_topic_size=15,
    nr_topics="auto",
    verbose=True
)


# ================================
# STEP 3 — FIT MODEL
# ================================
topics, probs = topic_model.fit_transform(docs_all)

topic_labels = topic_model.generate_topic_labels()


# ================================
# STEP 4 — OUTLIER REDUCTION
# ================================
print("\nReducing outliers...")

topics = topic_model.reduce_outliers(
    docs_all,
    topics,
    probabilities=probs
)


# ================================
# STEP 5 — SAVE OUTPUTS
# ================================
df_docs = pd.DataFrame({
    "text": docs_all,
    "topic": topics
}).reset_index(drop=True)

meta_df = pd.DataFrame(meta).reset_index(drop=True)

df_docs = pd.concat([df_docs, meta_df], axis=1)

topic_info = topic_model.get_topic_info()

topic_info.to_csv(out_dir / "topic_info.csv", index=False)
df_docs.to_csv(out_dir / "doc_topics.csv", index=False)

print("Saved topic_info.csv and doc_topics.csv")


# ================================
# STEP 6 — ANALYSIS
# ================================

# ---------- Topic → Emotion ----------
if "emotion" in df_docs.columns and df_docs["emotion"].notna().any():
    print("\nRunning Topic → Emotion Analysis...")

    topic_emotion = (
        df_docs.groupby(["topic", "emotion"])
        .size()
        .unstack(fill_value=0)
    )

    topic_emotion_norm = topic_emotion.div(topic_emotion.sum(axis=1), axis=0)

    topic_emotion_norm.to_csv(out_dir / "topic_emotion_distribution.csv")

    entropy = -(topic_emotion_norm * np.log(topic_emotion_norm + 1e-9)).sum(axis=1)
    entropy.to_csv(out_dir / "topic_emotion_entropy.csv")

    print("Saved topic-emotion analysis")


# ---------- Topic → Time ----------
if "time_period" in df_docs.columns and df_docs["time_period"].notna().any():
    print("\nRunning Topic → Time Analysis...")

    topic_time = (
        df_docs.groupby(["topic", "time_period"])
        .size()
        .unstack(fill_value=0)
    )

    topic_time_norm = topic_time.div(topic_time.sum(axis=1), axis=0)

    topic_time_norm.to_csv(out_dir / "topic_time_distribution.csv")

    print("Saved topic-time analysis")


# ================================
# STEP 7 — VISUALIZATIONS (AFTER LABELS)
# ================================
# fig = topic_model.visualize_barchart(top_n_topics=30)
# fig.write_html(out_dir / "barchart.html")
# fig2 = topic_model.visualize_topics()
# fig2.write_html(out_dir / "topics_viz.html")

fig = topic_model.visualize_barchart(
    topics=topic_info["Topic"].tolist(),
    custom_labels=True
)
fig.write_html(out_dir / "barchart.html")


fig2 = topic_model.visualize_topics()
fig2.write_html(out_dir / "topics_viz.html")

print("Saved visualizations")

# ================================
# STEP 8 — TOPIC NAMING
# ================================
def make_topic_name(keywords):
    words = [w for w, _ in keywords[:5]]

    if any("parking" in w for w in words):
        return "Parking & Transportation"

    if any("gpa" in w or "grade" in w for w in words):
        return "Grades & GPA"

    if any("housing" in w or "roommate" in w for w in words):
        return "Housing & Roommates"

    if any("club" in w or "friends" in w for w in words):
        return "Clubs & Social Life"

    if any("admission" in w or "transfer" in w for w in words):
        return "Admissions & Transfers"

    if any("commencement" in w or "graduation" in w for w in words):
        return "Graduation & Commencement"

    if any("ticket" in w for w in words):
        return "Tickets & Events"

    return "Other Student Discussions"


# Build labels
topic_labels = {}

for topic_id in topic_model.get_topic_info()["Topic"]:
    if topic_id == -1:
        continue

    keywords = topic_model.get_topic(topic_id)
    topic_labels[topic_id] = make_topic_name(keywords)


# APPLY ONCE (clean)
topic_model.set_topic_labels(topic_labels)

# Apply to dataframe
df_docs["topic_name"] = df_docs["topic"].map(topic_labels)

print("Topic labeling complete")