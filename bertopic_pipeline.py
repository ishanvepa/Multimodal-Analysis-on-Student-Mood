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
school = "UNC"
out_dir = Path(f"bertopic_outputs_{school}")
out_dir.mkdir(exist_ok=True)

file_source = f"cleaned_reddit_reviews/cleaned_redditreviews_{school}.json.zip"


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

    text = re.sub(r"http\S+|www\S+", "", text)
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
vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer,
    min_topic_size=15,
    nr_topics="auto",
    verbose=True
)


topics, probs = topic_model.fit_transform(docs_all)

topics = topic_model.reduce_outliers(docs_all, topics, probabilities=probs)


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