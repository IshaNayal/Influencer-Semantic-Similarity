import os
from collections.abc import Iterator

import pandas as pd
from datasets import load_dataset


def _ensure_dataframe(df_like):
    if isinstance(df_like, pd.DataFrame):
        return df_like
    if isinstance(df_like, Iterator):
        frames = list(df_like)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(df_like)

def get_deepika_posts():
    print("Loading dataset...")
    try:
        # Try loading from Hugging Face
        hf_dataset = load_dataset("ishanayal16/influencer_data")
        df = _ensure_dataframe(hf_dataset["train"].to_pandas())
    except Exception as e:
        print(f"Error loading from HF: {e}")
        if os.path.exists("organic_data.csv"):
            df = _ensure_dataframe(pd.read_csv("organic_data.csv"))
        else:
            print("No dataset found.")
            return

    # Normalize column names if needed
    if 'Name_x' in df.columns:
        df['influencer'] = df['Name_x']
    
    target_influencer = 'deepika.padukone.the.princess'
    print(f"Filtering for {target_influencer}...")
    
    subset = df[df['influencer'] == target_influencer]
    
    # Try to find sponsored posts
    # 1. Check 'Sponsored' column if exists
    if 'Sponsored' in df.columns:
        sponsored_posts = subset[subset['Sponsored'] == 1]
    else:
        sponsored_posts = pd.DataFrame()
    
    # 2. Check keywords if not enough
    if len(sponsored_posts) < 10:
        keywords = ['#ad', '#sponsored', '#partner', '#gifted', '#collab', 'promotion', 'brand', 'ambassador', 'loreal', 'asianpaints', 'tanishq', 'oppo', 'epigamia', 'nescafe']
        pattern = '|'.join(keywords)
        keyword_matches = subset[subset['caption'].str.contains(pattern, case=False, na=False)]
        sponsored_posts = pd.concat([sponsored_posts, keyword_matches]).drop_duplicates()
    
    # Get top 10
    posts = sponsored_posts.head(10)['caption'].tolist()
    
    # If still less than 10, fill with others (but label them)
    if len(posts) < 10:
        remaining = 10 - len(posts)
        others = subset[~subset.index.isin(sponsored_posts.index)].head(remaining)['caption'].tolist()
        posts.extend(others)

    print(f"\nFound {len(posts)} posts for testing:\n")
    
    with open("deepika_test_cases.txt", "w", encoding="utf-8") as f:
        for i, post in enumerate(posts, 1):
            # Clean up newlines for display
            clean_post = post.replace('\n', ' ')
            output = f"**Post {i}:**\n{clean_post}\n" + "-"*50 + "\n"
            # print(output) # Avoid printing to console due to encoding issues
            f.write(output + "\n")
    print("Saved posts to deepika_test_cases.txt")

if __name__ == "__main__":
    get_deepika_posts()
