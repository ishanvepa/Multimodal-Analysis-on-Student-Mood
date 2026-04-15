import pandas as pd

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("bertopic_outputs_UNC/dashboard_data.csv")

st.title("📊 Reddit Topic Explorer (BERTopic)")

# ================================
# SIDEBAR FILTERS
# ================================
topic_options = ["All"] + sorted(df["topic_name"].dropna().unique())
emotion_options = ["All"] + sorted(df["emotion"].dropna().unique())

selected_topic = st.sidebar.selectbox("Topic", topic_options)
selected_emotion = st.sidebar.selectbox("Emotion", emotion_options)


# ================================
# FILTER DATA
# ================================
filtered = df.copy()

if selected_topic != "All":
    filtered = filtered[filtered["topic_name"] == selected_topic]

if selected_emotion != "All":
    filtered = filtered[filtered["emotion"] == selected_emotion]


# ================================
# SUMMARY METRICS
# ================================
st.metric("Total Posts", len(filtered))
st.metric("Unique Topics", filtered["topic"].nunique())


# ================================
# TOPIC DISTRIBUTION
# ================================
st.subheader("📌 Topic Distribution")

topic_counts = filtered["topic_name"].value_counts()
st.bar_chart(topic_counts)


# ================================
# DATA TABLE
# ================================
st.subheader("📄 Posts")

st.dataframe(
    filtered[["text", "topic_name", "emotion", "time_period"]],
    use_container_width=True
)