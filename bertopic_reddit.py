import json
import random
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


file_source = "cleaned_reddit_reviews/cleaned_redditreviews_GATECH.json"
out_dir = "bertopic_outputs"
with open(file_source) as f:
    data = json.load(f)

# Keep only records with non-empty text
template = """\
{title}

{text}
"""
print(f"Total records loaded: {len(data)}")
docs_all = [template.format(title=d.get("Title", ""), text=d.get("Text", "")) for d in data if d.get("Text", "").strip()]
print(f"Total non-empty reviews: {len(docs_all)}")


'''
CountVectorizer is used to build a bag-of-words per cluster after clustering, 
it applies c-TF-IDF to extract the most representative words. 
The range of words to consider is set to ngrams. e.g. (1, 2) means unigrams and bigrams.
'''
vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))


'''
min_topic_size: minimum number of clusters
nr_topics: reduce initial number of clusters to specific number
top_n_words: number of words to represent each topic
'''
topic_model = BERTopic(
    vectorizer_model=vectorizer,
    # min_topic_size=10,
    # nr_topics="auto",
    nr_topics=30,
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs_all)

# ── Print topic overview ─────────────────────────────────────────
print("\n" + "=" * 60)
print("TOPIC OVERVIEW")
print("=" * 60)
topic_info = topic_model.get_topic_info()
print(topic_info.to_string(index=False))

print("\n" + "=" * 60)
print("TOP WORDS PER TOPIC")
print("=" * 60)
for topic_id in sorted(topic_model.get_topics().keys()):
    if topic_id == -1:
        continue
    words = [w for w, _ in topic_model.get_topic(topic_id)[:10]]
    print(f"Topic {topic_id}: {', '.join(words)}")

# ── Save outputs ─────────────────────────────────────────────────
# Topic info table
topic_info.to_csv(f"{out_dir}/topic_info.csv", index=False)

# Per-document topic assignments
df_docs = pd.DataFrame({"text": docs_all, "topic": topics})
df_docs.to_csv(f"{out_dir}/doc_topics.csv", index=False)
print(f"\nSaved: {out_dir}/topic_info.csv, {out_dir}/doc_topics.csv")

# Interactive HTML visualization (if running locally)
try:
    fig = topic_model.visualize_topics()
    fig.write_html(f"{out_dir}/topics_viz.html")
    print(f"\nSaved interactive topic map → {out_dir}/topics_viz.html")
except Exception as e:
    print(f"\nSkipped interactive viz: {e}")

# Bar chart of top words per topic
try:
    fig_bar = topic_model.visualize_barchart(top_n_topics=30)
    fig_bar.write_html(f"{out_dir}/barchart.html")
    print(f"Saved bar chart → {out_dir}/barchart.html")
except Exception as e:
    print(f"Skipped bar chart viz: {e}")

print("Done!")
