import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("Running GT vs UNC topic analysis pipeline...")

# =========================
# 1. LOAD AND MERGE DATA
# =========================

ROOT_DIR = Path(__file__).resolve().parent
schools = ["GATECH", "UNC"]

all_docs = []
topic_maps = []

for s in schools:
    # doc topics
    path = ROOT_DIR / f"bertopic_outputs_{s}" / "doc_topics.csv"
    docs = pd.read_csv(path)
    docs["school"] = s
    all_docs.append(docs)

    # topic names
    t = pd.read_csv(ROOT_DIR / f"bertopic_outputs_{s}" / "LLM_topics.csv")
    t["school"] = s
    topic_maps.append(t)

docs = pd.concat(all_docs, ignore_index=True)
topics = pd.concat(topic_maps, ignore_index=True)

# merge topic names
df = docs.merge(
    topics[["topic", "llm_name", "llm_description", "school"]],
    on=["topic", "school"],
    how="left"
)

print("Data merged ✔")

# =========================
# 2. LEADERBOARD (RAW TOPICS)
# =========================

leaderboard = df.groupby(
    ["school", "topic", "llm_name"]
).size().reset_index(name="count")

leaderboard.to_csv("leaderboard.csv", index=False)

print("Leaderboard saved ✔")

# =========================
# 3. TOPIC DIFFERENCE ANALYSIS
# =========================

pivot = leaderboard.pivot_table(
    index="llm_name",
    columns="school",
    values="count",
    fill_value=0
)

pivot["diff"] = pivot["GATECH"] - pivot["UNC"]

print("\nTOP GT-HEAVY TOPICS")
print(pivot.sort_values("diff", ascending=False).head(10))

print("\nTOP UNC-HEAVY TOPICS")
print(pivot.sort_values("diff").head(10))

# =========================
# 4. VISUALIZATION (RAW TOPICS)
# =========================

plot_data = pivot.drop(columns=["diff"]).copy()
plot_data["total"] = plot_data.sum(axis=1)
plot_data = plot_data.sort_values("total", ascending=False).head(15)
plot_data = plot_data.drop(columns=["total"])

plot_data.plot(kind="bar", figsize=(14,6))

plt.title("Top 15 Topics: GT vs UNC Comparison")
plt.ylabel("Number of Posts")
plt.xlabel("Topic")
plt.xticks(rotation=45, ha="right")
plt.legend(title="School")
plt.tight_layout()

plt.savefig("gt_unc_topics.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 5. THEME-LEVEL GROUPING (NEW — IMPORTANT)
# =========================

topic_map = {
    "Academic & Coursework": [
        "Academic Programs",
        "Academics & Course Planning",
        "Academic Affairs",
        "Computer Science Courses",
        "Study Habits & Time",
        "AI & Academic Integrity"
    ],

    "Housing & Campus Living": [
        "Housing & Parking",
        "Housing & Renovations",
        "Housing & Facilities"
    ],

    "Social Life & Culture": [
        "Social Life & Relationships",
        "Greek Life & Socializing",
        "Student Life",
        "Student Life & Culture",
        "Community & Culture",
        "Campus Traditions",
        "Anime & Fandom"
    ],

    "Athletics & Recreation": [
        "Athletics & Sports",
        "Sports & Athletics",
        "Campus Recreation & Fitness",
        "Administration & Athletics"
    ],

    "Safety & Health": [
        "Safety & Security",
        "Safety & Emergency Preparedness",
        "Health & Medical Care",
        "Health & Safety",
        "Student Health & Insurance",
        "Safety & Animal Incidents"
    ],

    "Transportation & Infrastructure": [
        "Campus Transportation",
        "Transportation & Transit"
    ],

    "Technology & Engineering": [
        "Information Technology",
        "Engineering & Design"
    ],

    "Career & Finance": [
        "Career & Employment",
        "Finance & Economics",
        "Personal Finance"
    ],

    "Campus Life Misc": [
        "Campus Pets",
        "Hair & Grooming",
        "Dining & Food",
        "Weather & Climate",
        "Film & Media",
        "Lost & Found",
        "Graduation Attire",
        "Campus Architecture"
    ],

    "Politics & Governance": [
        "Political Activism & Rallies",
        "Campus Politics",
        "Politics & Government",
        "Student Government",
        "Campus Politics & Safety"
    ]
}

def map_theme(topic):
    for theme, topics in topic_map.items():
        if topic in topics:
            return theme
    return "Other"

df["theme"] = df["llm_name"].apply(map_theme)

# =========================
# 6. THEME LEADERBOARD (CLEAN INSIGHT)
# =========================

theme_leaderboard = df.groupby(
    ["school", "theme"]
).size().reset_index(name="count")

theme_pivot = theme_leaderboard.pivot_table(
    index="theme",
    columns="school",
    values="count",
    fill_value=0
)

theme_pivot.plot(kind="bar", figsize=(12,6))

plt.title("GT vs UNC — High-Level Theme Comparison")
plt.ylabel("Number of Posts")
plt.xlabel("Theme")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("gt_unc_themes.png", dpi=300, bbox_inches="tight")
plt.show()

print("Theme-level visualization saved ✔")

# =========================
# 7. EMOTION ANALYSIS (SAFE FIX)
# =========================

if "emotion" in df.columns and df["emotion"].notna().any():

    print("\nEmotion column found ✔")

    emotion_df = df.groupby(["emotion", "llm_name"]).size().reset_index(name="count")

    if len(emotion_df) == 0:
        print("No emotion data to plot — skipping heatmap")
    else:
        pivot_emotion = emotion_df.pivot_table(
            index="emotion",
            columns="llm_name",
            values="count",
            fill_value=0
        )

        if pivot_emotion.shape[0] > 0 and pivot_emotion.shape[1] > 0:
            plt.figure(figsize=(12,6))
            sns.heatmap(pivot_emotion, cmap="coolwarm")
            plt.title("Emotion Distribution Across Topics")
            plt.tight_layout()
            plt.savefig("emotion_heatmap.png", dpi=300, bbox_inches="tight")
            plt.show()
        else:
            print("Empty pivot table — skipping heatmap")

else:
    print("\nNo valid emotion column — skipping emotion analysis")

# =========================
# DONE
# =========================

print("\nDone ✔")
print("Outputs:")
print("- leaderboard.csv")
print("- gt_unc_topics.png")
print("- gt_unc_themes.png")