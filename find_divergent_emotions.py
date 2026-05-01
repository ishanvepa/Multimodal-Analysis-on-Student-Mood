import zipfile
import json
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def load_data(zip_path):
    """Extracts the first JSON file found inside the zip."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        for f in z.namelist():
            if f.endswith('.json'):
                with z.open(f) as jf:
                    return json.load(jf)
    # Fallback if the file doesn't have a .json extension internally
    with zipfile.ZipFile(zip_path, 'r') as z:
        return json.load(z.open(z.namelist()[0]))

def get_emotions_df(data, school_name):
    """Extracts the 'Emotions' dictionary from each post into a DataFrame."""
    rows = []
    for d in data:
        emotions = d.get('Emotions', {})
        if emotions:
            emotions['School'] = school_name
            rows.append(emotions)
    return pd.DataFrame(rows)

def main():
    print("Loading datasets...")
    gatech_data = load_data('processed_reddit_reviews/processed_redditreviews_GATECH.json.zip')
    unc_data = load_data('processed_reddit_reviews/processed_redditreviews_UNC.zip')

    df_g = get_emotions_df(gatech_data, 'GATECH')
    df_u = get_emotions_df(unc_data, 'UNC')

    # Get the list of all 28 emotion names
    emotions = [c for c in df_g.columns if c != 'School']

    results = []
    print("Running Mann-Whitney U tests and calculating Cohen's d...")
    
    for e in emotions:
        g_vals = df_g[e].dropna().values
        u_vals = df_u[e].dropna().values
        
        if len(g_vals) == 0 or len(u_vals) == 0: 
            continue
        
        # Non-parametric test for difference in distributions
        stat, p = mannwhitneyu(g_vals, u_vals, alternative='two-sided')
        
        # Calculate Cohen's d for Effect Size
        n1, n2 = len(g_vals), len(u_vals)
        mean_diff = np.mean(g_vals) - np.mean(u_vals)
        pooled_sd = np.sqrt(((n1 - 1) * np.var(g_vals) + (n2 - 1) * np.var(u_vals)) / (n1 + n2 - 2))
        cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0
        
        results.append({
            'Emotion': e, 
            'P_Value': p, 
            'Cohens_D': cohens_d, 
            'Mean_GATECH': np.mean(g_vals), 
            'Mean_UNC': np.mean(u_vals)
        })

    # Sort the results by the absolute value of Cohen's D to find the most impactful differences
    res_df = pd.DataFrame(results).sort_values(by='Cohens_D', key=abs, ascending=False)
    
    print("\nTop Divergent Emotions (Sorted by Effect Size Magnitude):")
    print(res_df.head(10).to_string(index=False))
    
    # Optional: Save to CSV for the research paper appendix
    res_df.to_csv('emotion_divergence_stats.csv', index=False)
    print("\nFull statistics saved to emotion_divergence_stats.csv")

if __name__ == "__main__":
    main()
