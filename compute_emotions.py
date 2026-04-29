import json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent
IN_DIRS = [ROOT / "processed_reddit_reviews"]
OUT_DIR = ROOT / "emotions_output"
OUT_DIR.mkdir(exist_ok=True)


def find_json_files():
    files = []
    for d in IN_DIRS:
        if not d.exists():
            continue
        # include .json, .json.zip, and .zip files containing a single JSON
        files.extend(sorted(d.glob("*.json")))
        files.extend(sorted(d.glob("*.json.zip")))
        files.extend(sorted(d.glob("*.zip")))
    return files


def get_emotions(rec):
    # handle variants in key naming
    for k in ("Emotions", "emotion", "Emotion", "emotions"):
        if k in rec and isinstance(rec[k], dict):
            return rec[k]
    # try to find any key that maps to a dict of floats
    for k, v in rec.items():
        if isinstance(v, dict):
            return v
    return {}


def normalize_school(s):
    if not s:
        return None
    s = str(s).strip()
    if s.upper() in ("UNC", "UNIVERSITY OF NORTH CAROLINA", "UNC-CH", "UNC CHAPEL HILL"):
        return "UNC"
    if "GEORGIA" in s.upper() or "TECH" in s.upper() or s.upper() in ("GATECH", "GEORGIA TECH"):
        return "GATECH"
    return s


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def aggregate():
    files = find_json_files()
    campus_records = defaultdict(list)
    for p in files:
        try:
            if p.suffix == ".zip":
                # open zip and read first .json file inside
                import zipfile
                with zipfile.ZipFile(p, "r") as z:
                    json_files = [f for f in z.namelist() if f.endswith(".json")]
                    if not json_files:
                        continue
                    with z.open(json_files[0]) as fh:
                        data = json.load(fh)
            else:
                data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for rec in data:
            school = normalize_school(rec.get("School") or rec.get("school"))
            if school not in ("UNC", "GATECH"):
                continue
            campus_records[school].append(rec)

    summary_rows = []
    top1_counts = {"UNC": Counter(), "GATECH": Counter()}
    top2_counts = {"UNC": Counter(), "GATECH": Counter()}
    top3_counts = {"UNC": Counter(), "GATECH": Counter()}
    margins = {"UNC": [], "GATECH": []}

    for campus, recs in campus_records.items():
        for r in recs:
            emotions = get_emotions(r)
            if not emotions:
                continue
            items = []
            for k, v in emotions.items():
                fv = safe_float(v)
                if fv is None:
                    continue
                items.append((k, fv))
            if not items:
                continue
            items.sort(key=lambda kv: kv[1], reverse=True)
            labels = [k for k, _ in items]
            probs = [v for _, v in items]

            top1 = labels[0]
            top1_prob = probs[0]
            top2 = labels[1] if len(labels) >= 2 else None
            top2_prob = probs[1] if len(probs) >= 2 else 0.0
            top3 = labels[2] if len(labels) >= 3 else None

            top1_counts[campus][top1] += 1
            if top2:
                top2_counts[campus][f"{top1} + {top2}"] += 1
            if top3:
                top3_counts[campus][f"{top1} + {top2} + {top3}"] += 1

            margins[campus].append(top1_prob - top2_prob)

            summary_rows.append({
                "campus": campus,
                "top1": top1,
                "top1_prob": top1_prob,
                "top2": top2,
                "top2_prob": top2_prob,
                "top3": top3,
                "unique_id": r.get("Unique_ID") or r.get("UniqueId") or r.get("id")
            })

    # Save detailed per-document summary
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "doc_emotion_summary.csv", index=False)

    # Save aggregated counts and proportions
    def save_counts(counter_map, name, top_n=20):
        rows = []
        for campus, ctr in counter_map.items():
            total = sum(ctr.values())
            for label, cnt in ctr.most_common(top_n):
                rows.append({"campus": campus, "label": label, "count": cnt, "proportion": cnt / total if total else 0})
        pd.DataFrame(rows).to_csv(OUT_DIR / name, index=False)

    save_counts(top1_counts, "emotion_top1_by_campus.csv")
    save_counts(top2_counts, "emotion_top2_by_campus.csv")
    save_counts(top3_counts, "emotion_top3_by_campus.csv")

    # Plots
    sns.set(style="whitegrid")
    topn = 12
    for campus in ("GATECH", "UNC"):
        top = top1_counts[campus].most_common(topn)
        if not top:
            continue
        labels, counts = zip(*top)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(labels), palette="muted")
        plt.title(f"Top-{topn} Top-1 Emotions — {campus}")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"top1_{campus}.png", dpi=200)
        plt.close()

    # Margin distribution
    for campus in ("GATECH", "UNC"):
        if margins[campus]:
            plt.figure(figsize=(8, 4))
            sns.histplot(margins[campus], bins=50, kde=True)
            plt.title(f"Top1-Top2 Probability Margin — {campus}")
            plt.xlabel("Top1 - Top2")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"margin_{campus}.png", dpi=200)
            plt.close()

    print("Saved outputs to:", OUT_DIR)


if __name__ == "__main__":
    aggregate()
