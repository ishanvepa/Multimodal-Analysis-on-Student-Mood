import pandas as pd
import json
from pathlib import Path

# Load raw scraped JSON
input_file = "google_reviews_data/googlereviews_Georgia-Tech_part2.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data)

# Build cleaned dataframe
clean_df = pd.DataFrame({
    "Source": "Google Reviews",
    "Timestamp": df["publishedAtDate"],
    "Title": "",  # intentionally left empty but included
    "Text": df["text"].fillna(""),
    "Star_Rating": df["stars"],
    "Author": df["name"],
    "School": "Georgia Tech",
    "Location": "Klaus Advanced Computing Building",
    "Photo_Urls": df["reviewImageUrls"].apply(
        lambda x: x if isinstance(x, list) else []
    )
})

# Optional: remove rows with no text AND no images
# clean_df = clean_df[(clean_df["text"] != "") | (clean_df["photo_urls"].str.len() > 0)]

# Convert to JSON
output_dir = Path("cleaned_google_reviews")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "cleaned_googlereviews_Georgia-Tech_part2.json"
clean_df.to_json(output_file, orient="records", indent=2)

print(f"Saved cleaned dataset to {output_file}")