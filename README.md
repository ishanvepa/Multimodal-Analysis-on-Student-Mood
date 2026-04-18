### Google Review Images

Download images from [Google drive](https://drive.google.com/file/d/1rrGMIap6TmVyX9I59e9uPLgAMUw_2hhC/view?usp=sharing).  
Filename: G_{school}\_{index of the google review file}\_{index of the review in that file}#{index of the photo in that review}.ext = {Unique_ID}#{index of the photo in the review}.ext


### Place perception
```bash
pip install zensvi==1.4.7
```

in line 275 of `{your site-packages path}/zensvi/cv/classification/perception.py` add this line, otherwise, models will not be successfully loaded

```python
checkpoint_path = Path(__file__).parent.parent.parent.parent.parent / model_load_path / file_name
```
then run
```python
python place_perception.py
```

### Topic Modeling (BERTopic)

Now we use [BERTopic](https://maartengr.github.io/BERTopic/) to discover themes in Reddit posts from each school.
```bash
pip install bertopic
```

#### Planned analyses

- **Analysis 1 — Topics first, then emotions within each topic.** Fit one topic model on all posts, then look at the emotion distribution per topic.
  - Q1: Which topics are most strongly associated with anxiety, anger, sadness, etc.?
  - Q2: Are certain topics emotionally homogeneous, or emotionally mixed?
- **Analysis 2 — Emotions first, then topics within each emotion.** Split posts by emotion label, then fit a topic model per subset.
  - Q1: What major themes appear in Reddit posts expressing a given emotion?

#### How BERTopic works

Quoting the original paper: *"BERTopic generates topic representations through three steps. First, each document is converted to its embedding representation using a pre-trained language model. Then, before clustering these embeddings, the dimensionality of the resulting embeddings is reduced to optimize the clustering process. Lastly, from the clusters of documents, topic representations are extracted using a custom class-based variation of TF-IDF."*

The pipeline of BERTopic:

1. **Embed** each document with a pre-trained sentence transformer (the default `all-MiniLM-L6-v2` ).
2. **Reduce dimensionality** of the embeddings with UMAP so clustering is tractable.
3. **Cluster** the reduced embeddings with HDBSCAN. Each cluster becomes a candidate topic.
4. **Label topics** with a class-based TF-IDF (c-TF-IDF): for each cluster, `CountVectorizer` builds a bag-of-words, and c-TF-IDF picks the terms that are most distinctive for that cluster versus all others. This is how we get the keywords you see per topic.

#### Running the script

The script is [`bertopic_pipeline.py`]. Currently, I concatenate each post's `Title` and `Text`, and fits BERTopic on the full set of non-empty reviews.

Run:
```bash
python bertopic_pipeline.py UNC # or python bertopic_pipeline.py GATECH
```

Outputs:
- `bertopic_topic_info.csv` — one row per topic with size and top keywords
- `bertopic_doc_topics.csv` — each input document with its assigned topic
- `bertopic_topics_viz.html` — interactive topic distance map
- `bertopic_barchart.html` — top words per topic as a bar chart

#### Params worth tuning

All of these live in `bertopic_pipeline.py`:

- `CountVectorizer(stop_words="english", ngram_range=(1, 2))`
  - `stop_words`: removes common English words from the topic *labels* (not from clustering).
  - `ngram_range`: `(1, 2)` means topic keywords can be unigrams *or* bigrams (e.g. `"machine learning"` as a single term). Widen to `(1, 3)` for longer phrases.
- `BERTopic(...)` arguments:
  - `nr_topics`: final number of topics. Currently hard-coded to `30`; set to `"auto"` to let HDBSCAN decide, or pick a smaller/larger integer.
  - `min_topic_size`: minimum number of documents that can form a topic. Larger values → fewer, coarser topics. Smaller values → more, finer-grained topics (but also more noise).
  - `top_n_words`: how many keywords to keep per topic representation.
  - `embedding_model`: defaults to `all-MiniLM-L6-v2`. Swap to `all-mpnet-base-v2` for higher-quality embeddings at the cost of speed.

#### TODOs

1. **Text preprocessing.** Most of this is done in `bertopic_pipeline.py`:
  - Done:
    - `[deleted]` / `[removed]` posts filtered
    - AutoModerator / bot messages filtered via `BOT_PATTERNS`
    - URLs, markdown (`**bold**`, `>quotes`), `/u/user`, `/r/sub` stripped
    - HTML entities (`&amp;`, `&gt;`, `&#x200b;`) decoded
    - Very short posts (< 5 tokens) dropped
    - School-specific stopwords (`gatech`, `gt`, `georgia`, `tech`, `unc`, `tarheel`, …) via `SCHOOL_STOPWORDS`

  - TODO: **Near-duplicate deduplication** — reposts, crossposts, auto-replies still leak through and inflate certain clusters.
2. **LLM topic labeling.** Keyword lists are hard to read. Plan to feed each topic's keyword list + representative documents into an LLM (gpt-oss via Ollama) to synthesize a short, human-readable topic name (e.g. `["exam", "study", "final", "grade"]` → `"Exams and studying"`) and a one-sentence description.
