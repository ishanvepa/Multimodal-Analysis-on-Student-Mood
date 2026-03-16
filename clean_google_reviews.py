import pandas as pd
import json
from pathlib import Path

# Load raw scraped JSON
input_file = "google_reviews_data/studentcenter.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data)

# Convert timestamp to datetime
df["publishedAtDate"] = pd.to_datetime(df["publishedAtDate"], errors="coerce")


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

df["Time_Period"] = df["publishedAtDate"].apply(assign_time_period)


# -------- Build cleaned dataframe --------
clean_df = pd.DataFrame({
    "Source": "Google Reviews",
    "Timestamp": df["publishedAtDate"].dt.strftime("%Y-%m-%d %H:%M:%S"),
    "Title": "",
    "Text": df["text"].fillna(""),
    "Star_Rating": df["stars"],
    "Author": df["name"],
    "School": "Georgia Tech",
    "Location": "Student Center",
    "Photo_Urls": df["reviewImageUrls"].apply(
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
    create_unique_id(i + 1, clean_df.loc[i, "Source"], clean_df.loc[i, "School"])
    for i in range(len(clean_df))
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
output_dir = Path("cleaned_google_reviews")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "cleaned_googlereviews_GATECH_Student_Center.json"
clean_df.to_json(output_file, orient="records", indent=2)

print(f"Saved cleaned dataset to {output_file}")