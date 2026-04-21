import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

st.set_page_config(page_title="Reddit Topic Explorer", layout="wide")

st.title("📊 Reddit Topic Explorer (GT vs UNC BERTopic)")

# ================================
# LOAD DATA FUNCTION
# ================================
ROOT_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_school_data(school):
    docs = pd.read_csv(ROOT_DIR / f"bertopic_outputs_{school}" / "doc_topics.csv")
    topics = pd.read_csv(ROOT_DIR / f"bertopic_outputs_{school}" / "LLM_topics.csv")

    df = docs.merge(
        topics[["topic", "llm_name", "llm_description"]],
        on="topic",
        how="left"
    )

    df["school"] = school
    return df


# ================================
# SIDEBAR CONTROLS
# ================================
st.sidebar.header("🔧 Filters")

mode = st.sidebar.radio("Select View Mode", ["Single School", "Compare GT vs UNC"])
view_type = st.sidebar.radio("Analysis Type", ["Topics", "Themes"])


# ================================
# THEME MAPPING
# ================================
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
    "Transportation": [
        "Campus Transportation",
        "Transportation & Transit"
    ],
    "Tech & Engineering": [
        "Information Technology",
        "Engineering & Design"
    ],
    "Career & Finance": [
        "Career & Employment",
        "Finance & Economics",
        "Personal Finance"
    ],
    "Campus Misc": [
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


# ================================
# LOAD DATA
# ================================
if mode == "Single School":
    school = st.sidebar.selectbox("Select School", ["UNC", "GATECH"])
    df = load_school_data(school)

else:
    df_gt = load_school_data("GATECH")
    df_unc = load_school_data("UNC")
    df = pd.concat([df_gt, df_unc], ignore_index=True)


# ================================
# APPLY THEMES
# ================================
df["theme"] = df["llm_name"].apply(map_theme)


# ================================
# METRICS
# ================================
col1, col2, col3 = st.columns(3)

col1.metric("Total Posts", len(df))
col2.metric("Unique Topics", df["llm_name"].nunique())
col3.metric("Unique Themes", df["theme"].nunique())

st.divider()


# ================================
# VISUALIZATION
# ================================
st.subheader("📊 Distribution")

if view_type == "Topics":
    data = df["llm_name"].value_counts().head(15)

    fig, ax = plt.subplots()
    data.plot(kind="bar", ax=ax)
    ax.set_title("Top Topics")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)

else:
    data = df["theme"].value_counts()

    fig, ax = plt.subplots()
    data.plot(kind="bar", ax=ax)
    ax.set_title("Top Themes")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)


# ================================
# GT vs UNC COMPARISON (ONLY IN COMPARE MODE)
# ================================
if mode == "Compare GT vs UNC":
    st.subheader("⚖️ GT vs UNC Topic Comparison")

    # ----------------------------
    # Build comparison table
    # ----------------------------
    comp = (
        df.groupby(["llm_name", "school"])
        .size()
        .unstack()
        .reindex(columns=["GATECH", "UNC"], fill_value=0)
    )

    # keep top topics
    comp["total"] = comp.sum(axis=1)
    comp = comp.sort_values("total", ascending=False).head(15)
    comp = comp.drop(columns=["total"])

    # ----------------------------
    # Plot
    # ----------------------------
    x = np.arange(len(comp.index))
    width = 0.4

    fig, ax = plt.subplots(figsize=(18, 8))

    ax.bar(x - width/2, comp["GATECH"], width, label="GATECH")
    ax.bar(x + width/2, comp["UNC"], width, label="UNC")

    wrapped_labels = [textwrap.fill(label, 18) for label in comp.index]

    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels, rotation=0, ha="center")

    ax.set_title("Top 15 Topics: GT vs UNC Comparison")
    ax.set_xlabel("Topics")
    ax.set_ylabel("Number of Posts")

    ax.legend()
    ax.margins(x=0.05)

    plt.tight_layout()
    st.pyplot(fig)


# ================================
# DATA TABLE
# ================================
st.subheader("📄 Sample Posts")

display_cols = ["llm_name", "theme", "school"]
st.dataframe(df[display_cols].head(200), use_container_width=True)