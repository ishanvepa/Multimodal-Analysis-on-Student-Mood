import ast
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import textwrap

st.set_page_config(page_title="Reddit Topic Explorer", layout="wide")

st.title("📊 Reddit Topic Explorer (GT vs UNC BERTopic)")

# ================================
# LOAD DATA FUNCTION
# ================================
ROOT_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_school_data(school):
    base = ROOT_DIR / f"bertopic_outputs_{school}"
    docs = pd.read_csv(base / "doc_topics.csv")
    llm = pd.read_csv(base / "LLM_topics.csv")
    info = pd.read_csv(base / "topic_info.csv").rename(columns={"Topic": "topic"})

    meta = llm.merge(
        info[["topic", "Representative_Docs"]],
        on="topic", how="left",
    )

    df = docs.merge(
        meta[["topic", "llm_name", "llm_description", "keywords", "Representative_Docs"]],
        on="topic", how="left",
    )

    df["school"] = school
    return df


def _parse_docs(cell):
    if pd.isna(cell):
        return []
    try:
        v = ast.literal_eval(cell)
        return v if isinstance(v, list) else [str(v)]
    except (ValueError, SyntaxError):
        return []


def _truncate(text, limit=220):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def build_hover(keywords_cell, repr_docs_cell, n_kw=5, n_docs=2):
    kws = [k.strip() for k in str(keywords_cell).split(",") if k.strip()][:n_kw]
    docs = _parse_docs(repr_docs_cell)[:n_docs]

    # Wrap keywords nicely
    kw_text = ", ".join(kws)
    kw_text = "<br>".join(textwrap.wrap(kw_text, width=40))

    parts = [f"<b>Top keywords:</b><br>{kw_text or '—'}"]

    for i, d in enumerate(docs, 1):
        wrapped = "<br>".join(textwrap.wrap(_truncate(d), width=50))
        parts.append(f"<b>[{i}]</b> {wrapped}")

    return "<br><br>".join(parts)


def topic_agg(df, school=None):
    """One row per llm_name with count + hover text."""
    if school is not None:
        df = df[df["school"] == school]
    agg = (
        df.groupby("llm_name", dropna=False)
        .agg(
            count=("topic", "size"),
            keywords=("keywords", "first"),
            repr_docs=("Representative_Docs", "first"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    agg["hover"] = agg.apply(
        lambda r: build_hover(r["keywords"], r["repr_docs"]), axis=1
    )
    return agg


def make_pie(agg, title):
    fig = go.Figure(
        go.Pie(
            labels=agg["llm_name"],
            values=agg["count"],
            hovertext=agg["hover"],
            hoverinfo="text",
            textinfo="percent+label",
        )
    )

    fig.update_layout(
        title=title,
        height=550,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            align="left",
            namelength=0  # prevents truncation weirdness
        )
    )

    return fig
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
st.caption("Hover a bar to see the top keywords and two representative posts.")

TOP_N = 15

if view_type == "Topics":

    # ----------------------------
    # SINGLE SCHOOL
    # ----------------------------
    if mode == "Single School":
        agg = topic_agg(df).head(TOP_N)

        # ========================
        # 1. BAR CHART (FIRST)
        # ========================
        st.subheader("📊 Top Topics")

        fig_bar = go.Figure(
            go.Bar(
                x=agg["llm_name"],
                y=agg["count"],
                hovertext=agg["hover"],
                hoverinfo="text",
                marker_color="steelblue",
            )
        )

        fig_bar.update_layout(
            title=f"Top {TOP_N} Topics — {school}",
            yaxis_title="Number of Posts",
            xaxis_tickangle=-40,
            height=500,
            hoverlabel=dict(bgcolor="white", font_size=12),
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # ========================
        # 2. PIE CHART (SECOND)
        # ========================
        st.subheader("🥧 Topic Distribution")

        fig_pie = make_pie(agg, f"{school} Topic Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)



    # ----------------------------
    # COMPARE MODE
    # ----------------------------
    elif mode == "Compare GT vs UNC":

        # ========================
        # 1. BAR CHART (FIRST)
        # ========================
        st.subheader("📊 Topic Count Comparison")

        agg_gt = topic_agg(df, "GATECH")
        agg_unc = topic_agg(df, "UNC")

        top_names = df["llm_name"].value_counts().head(TOP_N).index.tolist()

        gt = agg_gt.set_index("llm_name").reindex(top_names).reset_index()
        unc = agg_unc.set_index("llm_name").reindex(top_names).reset_index()

        gt[["count", "hover"]] = gt[["count", "hover"]].fillna({"count": 0, "hover": "(no posts)"})
        unc[["count", "hover"]] = unc[["count", "hover"]].fillna({"count": 0, "hover": "(no posts)"})

        fig = go.Figure()
        fig.add_bar(
            name="GATECH",
            x=gt["llm_name"],
            y=gt["count"],
            hovertext=gt["hover"],
            hoverinfo="text",
            marker_color="#B3A369",
        )
        fig.add_bar(
            name="UNC",
            x=unc["llm_name"],
            y=unc["count"],
            hovertext=unc["hover"],
            hoverinfo="text",
            marker_color="#4B9CD3",
        )

        fig.update_layout(
            barmode="group",
            title=f"Top {TOP_N} Topics — GT vs UNC",
            yaxis_title="Number of Posts",
            xaxis_tickangle=-40,
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # ========================
        # 2. PIE CHARTS (SECOND)
        # ========================
        st.subheader("🥧 Topic Distribution")

        agg_gt_top = topic_agg(df, "GATECH").head(TOP_N)
        agg_unc_top = topic_agg(df, "UNC").head(TOP_N)

        st.markdown("### GATECH")
        st.plotly_chart(make_pie(agg_gt_top, "GATECH"), use_container_width=True)

        st.markdown("### UNC")
        st.plotly_chart(make_pie(agg_unc_top, "UNC"), use_container_width=True)

        st.markdown(
            """
            <style>
            .js-plotly-plot .hoverlayer .hovertext {
                max-width: 300px !important;
                white-space: normal !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# ================================
# DATA TABLE
# ================================
st.subheader("📄 Sample Posts")

display_cols = ["llm_name", "theme", "school"]
st.dataframe(df[display_cols].head(200), use_container_width=True)