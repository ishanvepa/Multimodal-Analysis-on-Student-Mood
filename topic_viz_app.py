import ast
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import textwrap
import os

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
view_type = st.sidebar.radio("Analysis Type", ["Topics", "Themes", "Emotions"]) 


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

    # Emotion Analysis moved to a dedicated Analysis Type (see below)



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
# ================================
# EMOTION ANALYSIS
if view_type == "Emotions":
    emotion_dir = ROOT_DIR / "emotions_output"
    st.subheader("🙂 Emotion Analysis")

    if not emotion_dir.exists():
        st.info("No `emotions_output` directory found. Run `compute_emotions.py` to generate outputs.")
    else:
        if mode == "Single School":
            st.markdown(f"**Single School — {school}**")

            # Top-1 CSV filtered to selected school
            top1_csv = emotion_dir / "emotion_top1_by_campus.csv"
            if top1_csv.exists():
                em1 = pd.read_csv(top1_csv)
                em_school = em1[em1["campus"] == school].sort_values("proportion", ascending=False)
                fig = go.Figure(go.Bar(x=em_school["label"], y=em_school["proportion"], marker_color=("#4B9CD3" if school=="UNC" else "#B3A369")))
                fig.update_layout(title=f"Top-1 Emotion Proportions — {school}", xaxis_tickangle=-45, height=480)
                st.plotly_chart(fig, use_container_width=True)

            # show pre-generated images for the school if present
            img = emotion_dir / f"top1_{school}.png"
            marg = emotion_dir / f"margin_{school}.png"
            cols = st.columns(2)
            if img.exists():
                cols[0].image(str(img), width="stretch", caption=f"{school} — Top-1 Emotions")
            if marg.exists():
                cols[1].image(str(marg), width="stretch", caption=f"{school} — Top1–Top2 Margin")

            # show top-2/top-3 tables filtered to the school
            top2_csv = emotion_dir / "emotion_top2_by_campus.csv"
            top3_csv = emotion_dir / "emotion_top3_by_campus.csv"
            if top2_csv.exists():
                st.markdown("**Top-2 Emotion Pairs (counts & proportions)**")
                st.dataframe(pd.read_csv(top2_csv).query("campus == @school").sort_values("count", ascending=False).head(50))
            if top3_csv.exists():
                st.markdown("**Top-3 Emotion Triples (counts & proportions)**")
                st.dataframe(pd.read_csv(top3_csv).query("campus == @school").sort_values("count", ascending=False).head(50))

        else:  # Compare GT vs UNC
            st.markdown("**Compare — GATECH vs UNC**")

            top1_csv = emotion_dir / "emotion_top1_by_campus.csv"
            if top1_csv.exists():
                em1 = pd.read_csv(top1_csv)
                try:
                    pivot = em1.pivot(index="label", columns="campus", values="proportion").fillna(0)
                    fig_em = go.Figure()
                    if "GATECH" in pivot.columns:
                        fig_em.add_bar(name="GATECH", x=pivot.index, y=pivot["GATECH"].values, marker_color="#B3A369")
                    if "UNC" in pivot.columns:
                        fig_em.add_bar(name="UNC", x=pivot.index, y=pivot["UNC"].values, marker_color="#4B9CD3")
                    fig_em.update_layout(barmode="group", title="Top-1 Emotion Proportions — GT vs UNC", xaxis_tickangle=-45, height=520)
                    st.plotly_chart(fig_em, use_container_width=True)
                except Exception:
                    st.write("Could not render Top-1 proportions chart — CSV may be malformed.")

            # side-by-side pre-generated images
            img_gt = emotion_dir / "top1_GATECH.png"
            img_unc = emotion_dir / "top1_UNC.png"
            if img_gt.exists() and img_unc.exists():
                st.markdown("**Top-1 Emotion Bar Charts (pre-generated)**")
                cols = st.columns(2)
                cols[0].image(str(img_gt), width="stretch", caption="GATECH — Top-1 Emotions")
                cols[1].image(str(img_unc), width="stretch", caption="UNC — Top-1 Emotions")

            # margin histograms
            marg_gt = emotion_dir / "margin_GATECH.png"
            marg_unc = emotion_dir / "margin_UNC.png"
            if marg_gt.exists() and marg_unc.exists():
                st.markdown("**Top1–Top2 Margin (confidence)**")
                cols = st.columns(2)
                cols[0].image(str(marg_gt), width="stretch", caption="GATECH margin")
                cols[1].image(str(marg_unc), width="stretch", caption="UNC margin")

            # raw tables for top-2 / top-3 (full)
            top2_csv = emotion_dir / "emotion_top2_by_campus.csv"
            top3_csv = emotion_dir / "emotion_top3_by_campus.csv"
            if top2_csv.exists():
                st.markdown("**Top-2 Emotion Pairs (counts & proportions)**")
                st.dataframe(pd.read_csv(top2_csv).sort_values(["campus", "count"], ascending=[True, False]).head(200))
            if top3_csv.exists():
                st.markdown("**Top-3 Emotion Triples (counts & proportions)**")
                st.dataframe(pd.read_csv(top3_csv).sort_values(["campus", "count"], ascending=[True, False]).head(200))

# DATA TABLE
# ================================
st.subheader("📄 Sample Posts")

display_cols = ["llm_name", "theme", "school"]
st.dataframe(df[display_cols].head(200), use_container_width=True)