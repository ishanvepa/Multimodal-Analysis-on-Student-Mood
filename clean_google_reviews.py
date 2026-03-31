import pandas as pd
import json
from pathlib import Path

# Load raw scraped JSON

input_file = "reddit_reviews_data/redditreviews_UNC.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data)

# Convert timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")


# -------- Time Period Function --------
# Time period assignment
def assign_time_period(date):
    if pd.isna(date):
        return None

    month = date.month

    # Fall semester
    if month in [8, 9, 10, 11, 12]:
        if month in [9, 10]:
            return "Fall Midterms"
        elif month in [11, 12]:
            return "Fall Finals"
        else:
            return "Fall Semester"

    # Spring semester
    elif month in [1, 2, 3, 4, 5]:
        if month in [3, 4]:
            return "Spring Midterms"
        elif month in [5]:
            return "Spring Finals"
        else:
            return "Spring Semester"

    return "Other"

df["Time_Period"] = df["Timestamp"].apply(assign_time_period)


# -------- Build cleaned dataframe --------
# Clean empty text FIRST
df = df[df["Text"].notna() & (df["Text"].str.strip() != "")]
df = df.reset_index(drop=True)

clean_df = pd.DataFrame({
    "Source": df["Source"],
    "Timestamp": df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
    "Title": df["Title"].fillna(""),
    "Text": df["Text"].fillna(""),   
    "Star_Rating": None,
    "Author": df["Author"],          
    "School": "UNC",
    "Location": df["Location"],     
    "Photo_Urls": df["Photo_Urls"].apply(
        lambda x: x if isinstance(x, list) else []
    ),
    "Time_Period": df["Time_Period"]
})

# -------- Unique ID Creation --------
def create_unique_id(row_index, source, school):
    source_code = "G" if source == "Google Reviews" else "R"
    school_code = "GT" if school == "Georgia Tech" else "UNC"
    return f"{source_code}_{school_code}_{row_index:05d}"

clean_df["Unique_ID"] = [
    create_unique_id(i + 1, row["Source"], row["School"])
    for i, row in clean_df.iterrows()
]

# Reorder columns
clean_df = clean_df[[
    "Unique_ID",
    "Source",
    "Timestamp",
    "Time_Period",
    "Title",
    "Text",
    "Star_Rating",
    "Author",
    "School",
    "Location",
    "Photo_Urls"
]]

# Save JSON
output_dir = Path("cleaned_reddit_reviews")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "cleaned_redditreviews_UNC.json"
clean_df.to_json(output_file, orient="records", indent=2)

print(f"Saved cleaned dataset to {output_file}")