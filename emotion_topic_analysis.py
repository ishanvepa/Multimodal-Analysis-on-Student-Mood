"""
Topic x Emotion analysis (posts-only BERTopic + multi-label share(prob>0.3)).

For each school, on the BERTopic-filtered subset only:
  1. Justify emotion selection by plotting frequency before/after dropping
     neutral, under two definitions: argmax-per-doc and prob>0.3 (multi-label).
  2. Build topic x emotion matrices using share(prob > 0.3) — the fraction of
     a topic's docs where the emotion fires above threshold. Expressed as
     log10(lift) vs the corpus-mean baseline so over- and under-represented
     emotions are symmetric around 0.
  3. Eyeball: for each (topic, emotion) pair with lift >= 1.5 or <= 0.5, dump
     the topic's representative docs plus 3 random sampled docs from that
     topic into a text file for manual inspection.

Outputs land in emotion_topic_analysis/.
"""
from pathlib import Path
import ast
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(0)

SCHOOLS = ["UNC", "GATECH"]
PROB_THRESH = 0.3
TOP_K_EMOTIONS = 10   # per school; we then union across schools
MIN_TOPIC_SIZE = 200  # drop micro-topics — LLM labels are noisy and lift is unstable

OUT = Path("emotion_topic_analysis")
OUT.mkdir(exist_ok=True)
BT_DIR = lambda s: f"bertopic_outputs_{s}"


# ---------------------------------------------------------------------------
# Load: filtered subset (doc_topics) joined with per-doc Emotions on Unique_ID
# ---------------------------------------------------------------------------
def load_school(school):
    dt = pd.read_csv(f"{BT_DIR(school)}/doc_topics.csv")
    dt = dt[dt["topic"] != -1].copy()  # drop BERTopic noise bucket

    llm = pd.read_csv(f"{BT_DIR(school)}/LLM_topics.csv")
    name_map = dict(zip(llm["topic"], llm["llm_name"].astype(str).str.strip()))
    coherence_map = (
        dict(zip(llm["topic"], llm["coherence"].astype(str).str.strip()))
        if "coherence" in llm.columns else {}
    )

    def _decorate(t):
        c = coherence_map.get(t, "")
        return f"[{c}] " if c else ""
    name_map = {t: f"{_decorate(t)}{n}" for t, n in name_map.items()}

    raw = json.load(open(f"processed_reddit_reviews/processed_redditreviews_{school}.json"))
    em_by_id = {r["Unique_ID"]: r.get("Emotions") or {} for r in raw}

    rows = []
    for uid, topic in zip(dt["Unique_ID"], dt["topic"]):
        em = em_by_id.get(uid)
        if not em:
            continue
        rows.append((uid, topic, em))

    uids, topics, emotions = zip(*rows)
    em_df = pd.DataFrame(list(emotions)).fillna(0.0)
    em_df.insert(0, "Unique_ID", uids)
    em_df.insert(1, "topic", topics)
    sizes = em_df["topic"].value_counts().to_dict()
    em_df["topic_name"] = em_df["topic"].map(
        lambda t: f"{t}: {name_map.get(t, '?')} (n={sizes.get(t, 0):,})"
    )
    return em_df, name_map


def emotion_columns(df):
    return [c for c in df.columns if c not in {"Unique_ID", "topic", "topic_name"}]


# ---------------------------------------------------------------------------
# Plot 1: emotion frequency distributions (justify selection)
# ---------------------------------------------------------------------------
def plot_emotion_distribution(em_df, school, out_path):
    em_cols = emotion_columns(em_df)
    arr = em_df[em_cols].to_numpy()
    n = len(em_df)

    argmax_labels = np.array(em_cols)[arr.argmax(axis=1)]
    argmax_share = pd.Series(argmax_labels).value_counts().reindex(em_cols, fill_value=0) / n
    thresh_share = pd.Series((arr > PROB_THRESH).sum(axis=0), index=em_cols) / n

    # 2x2: rows = metric (argmax / prob>0.3); cols = with neutral / without neutral
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    metrics = [
        ("argmax (one label per doc)", argmax_share),
        (f"prob > {PROB_THRESH} (multi-label)", thresh_share),
    ]
    for row, (title, series) in enumerate(metrics):
        for col, drop_neutral in enumerate([False, True]):
            ax = axes[row][col]
            s = series.drop("neutral") if drop_neutral else series
            s = s.sort_values(ascending=False)
            colors = ["#cccccc" if e == "neutral" else "#4c72b0" for e in s.index]
            ax.bar(range(len(s)), s.values, color=colors)
            ax.set_xticks(range(len(s)))
            ax.set_xticklabels(s.index, rotation=60, ha="right", fontsize=8)
            ax.set_ylabel("share of docs")
            sub = "after dropping neutral" if drop_neutral else "all 28 GoEmotions labels"
            ax.set_title(f"{title} — {sub}")
            for i, (lbl, v) in enumerate(s.items()):
                if i < 3:
                    ax.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        f"{school} — Emotion frequency on BERTopic-filtered subset (n={n:,})\n"
        f"Top row: argmax. Bottom row: multi-label thresholded. "
        f"Right column drops neutral to expose secondary signal.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return thresh_share


# ---------------------------------------------------------------------------
# Plot 2: topic x emotion lift heatmap (share(prob > 0.3))
# ---------------------------------------------------------------------------
def topic_emotion_matrix(em_df, emotions):
    binarized = (em_df[emotions] > PROB_THRESH).astype(float)
    binarized["topic"] = em_df["topic"].values
    binarized["topic_name"] = em_df["topic_name"].values
    per_topic = binarized.groupby(["topic", "topic_name"])[emotions].mean()
    baseline = binarized[emotions].mean()
    baseline = baseline.replace(0, np.nan)
    lift = per_topic.div(baseline, axis=1)
    return per_topic, baseline, lift


def plot_lift_heatmap(per_topic, baseline, lift, topic_sizes, school, out_path):
    log_lift = np.log10(lift.replace([np.inf, -np.inf], np.nan)).clip(-1, 1)
    log_lift = log_lift.reset_index().set_index("topic_name").drop(columns=["topic"])

    order = topic_sizes.sort_values(ascending=False).index.tolist()
    log_lift = log_lift.reindex(order)

    em_order = baseline.sort_values(ascending=False).index.tolist()
    log_lift = log_lift[em_order]

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(em_order)),
                                    max(8, 0.32 * len(log_lift))))
    sns.heatmap(
        log_lift,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cbar_kws={"label": "log10(lift) — red=over, blue=under"},
        ax=ax,
        linewidths=0.3, linecolor="white",
    )
    ax.set_title(
        f"{school} — Topic x Emotion association (share(prob>{PROB_THRESH}))\n"
        f"Lift = topic value / corpus baseline; 0 = corpus-average; "
        f"+1 = 10x; -1 = 0.1x (clipped at +-1)",
        fontsize=11,
    )
    ax.set_xlabel("emotion")
    ax.set_ylabel("topic")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Eyeball: dump (topic, emotion) pairs with extreme lift to a text file
# ---------------------------------------------------------------------------
def dump_extreme_pairs(em_df, lift, school, out_path,
                       lift_high=1.5, lift_low=0.5, n_random=3):
    raw = json.load(open(f"processed_reddit_reviews/processed_redditreviews_{school}.json"))
    text_by_id = {
        r["Unique_ID"]: ((r.get("Title") or "") + " | " + (r.get("Text") or "")).strip(" |")
        for r in raw
    }
    info = pd.read_csv(f"{BT_DIR(school)}/topic_info.csv")
    rep_map = {}
    for t, reps in zip(info["Topic"], info["Representative_Docs"]):
        try:
            rep_map[int(t)] = ast.literal_eval(reps) if isinstance(reps, str) else []
        except Exception:
            rep_map[int(t)] = []

    docs_by_topic = em_df.groupby("topic")["Unique_ID"].apply(list).to_dict()

    rows = []
    for (tid, tname), emo_row in lift.iterrows():
        for emo, val in emo_row.items():
            if pd.isna(val) or val == 0:
                continue
            if val >= lift_high or val <= lift_low:
                rows.append((tid, tname, emo, val))
    rows.sort(key=lambda r: -abs(np.log10(r[3])))

    with open(out_path, "w") as f:
        f.write(f"# {school} — extreme (topic, emotion) pairs (share(prob>{PROB_THRESH}))\n")
        f.write(f"# lift >= {lift_high} or <= {lift_low}\n")
        f.write(f"# {len(rows)} pairs total\n\n")
        for tid, tname, emo, val in rows:
            direction = "OVER" if val >= lift_high else "UNDER"
            f.write("=" * 90 + "\n")
            f.write(f"[{direction}] topic={tname}  emotion={emo}  lift={val:.2f}\n")
            f.write("-- representative docs --\n")
            for d in (rep_map.get(tid) or [])[:2]:
                f.write(f"  * {d[:400]}\n")
            f.write(f"-- {n_random} random sampled docs from this topic --\n")
            pool = docs_by_topic.get(tid, [])
            sample_uids = random.sample(pool, min(n_random, len(pool)))
            for uid in sample_uids:
                t = text_by_id.get(uid, "")
                f.write(f"  [{uid}] {t[:400]}\n")
            f.write("\n")
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    loaded = {}
    thresh_shares = {}
    for s in SCHOOLS:
        em_df, _ = load_school(s)
        loaded[s] = em_df
        print(f"{s}: {len(em_df):,} docs joined to emotions, "
              f"{em_df['topic'].nunique()} topics")
        thresh_shares[s] = plot_emotion_distribution(
            em_df, s, OUT / f"emotion_distribution_{s}.png"
        )

    # Unified emotion set: top-K by prob>0.3 share in each school, union them,
    # drop neutral so the analysis isn't dominated by it.
    unified = set()
    for s in SCHOOLS:
        unified |= set(thresh_shares[s].drop("neutral").sort_values(
            ascending=False).head(TOP_K_EMOTIONS).index)
    unified = sorted(unified)
    print(f"\nUnified emotion set ({len(unified)}): {unified}")

    summary_rows = []
    for s in SCHOOLS:
        for e in thresh_shares[s].index:
            summary_rows.append({
                "school": s,
                "emotion": e,
                "share_prob_gt_0.3": thresh_shares[s][e],
                "in_unified_set": e in unified,
            })
    pd.DataFrame(summary_rows).to_csv(OUT / "emotion_selection_summary.csv", index=False)

    for s in SCHOOLS:
        em_df = loaded[s]
        sizes_by_id = em_df["topic"].value_counts()
        keep_ids = set(sizes_by_id[sizes_by_id >= MIN_TOPIC_SIZE].index)
        big = em_df[em_df["topic"].isin(keep_ids)]
        topic_sizes_named = big.groupby("topic_name")["topic"].count()
        print(f"  {s}: kept {len(keep_ids)} topics with >={MIN_TOPIC_SIZE} docs "
              f"(of {em_df['topic'].nunique()})")

        per_topic, baseline, lift = topic_emotion_matrix(big, unified)
        lift.to_csv(OUT / f"topic_emotion_lift_{s}.csv")
        per_topic.to_csv(OUT / f"topic_emotion_value_{s}.csv")
        plot_lift_heatmap(
            per_topic, baseline, lift, topic_sizes_named, s,
            OUT / f"topic_emotion_lift_{s}.png",
        )
        n_extreme = dump_extreme_pairs(
            big, lift, s,
            OUT / f"extreme_pairs_{s}.txt",
        )
        print(f"  wrote lift matrix + heatmap for {s} "
              f"({n_extreme} extreme pairs to extreme_pairs_{s}.txt)")

    print(f"\nAll outputs in {OUT}/")


if __name__ == "__main__":
    main()
