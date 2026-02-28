import pandas as pd
from datasets import load_dataset
import os

def extract_test_cases():
    print("Loading dataset...")
    try:
        # Try loading from Hugging Face
        hf_dataset = load_dataset("ishanayal16/influencer_data")
        df = hf_dataset['train'].to_pandas()
        print(f"Loaded {len(df)} posts from Hugging Face.")
    except Exception as e:
        print(f"Error loading from HF: {e}")
        if os.path.exists("organic_data.csv"):
            print("Falling back to local organic_data.csv")
            df = pd.read_csv("organic_data.csv")
        else:
            print("No dataset found.")
            return

    # Normalize column names if needed
    if 'Name_x' in df.columns:
        df['influencer'] = df['Name_x']
    
    # Identify top 5 influencers
    top_influencers = df['influencer'].value_counts().head(5).index.tolist()
    print(f"Top 5 Influencers: {top_influencers}\n")

    for influencer in top_influencers:
        print(f"### Influencer: {influencer}")
        
        # Filter for this influencer
        subset = df[df['influencer'] == influencer]
        
        # Try to find sponsored posts first
        # Assuming 'Sponsored' column exists and has 1 for sponsored
        if 'Sponsored' in df.columns:
            sponsored_posts = subset[subset['Sponsored'] == 1]
        else:
            sponsored_posts = pd.DataFrame()
        
        # If not enough labeled sponsored posts, look for keywords
        if len(sponsored_posts) < 5:
            keywords = ['#ad', '#sponsored', '#partner', '#gifted', '#collab']
            pattern = '|'.join(keywords)
            keyword_matches = subset[subset['caption'].str.contains(pattern, case=False, na=False)]
            sponsored_posts = pd.concat([sponsored_posts, keyword_matches]).drop_duplicates()
        
        # If still not enough, just take random posts (fallback)
        if len(sponsored_posts) < 5:
            remaining_needed = 5 - len(sponsored_posts)
            others = subset[~subset.index.isin(sponsored_posts.index)].head(remaining_needed)
            sponsored_posts = pd.concat([sponsored_posts, others])
        
        # Take top 5
        test_cases = sponsored_posts.head(5)['caption'].tolist()
        
        with open("test_cases.txt", "a", encoding="utf-8") as f:
            f.write(f"\n### Influencer: {influencer}\n")
            for i, post in enumerate(test_cases, 1):
                # Truncate for display if too long, but keep enough context
                display_text = post[:200] + "..." if len(post) > 200 else post
                display_text = display_text.replace('\n', ' ')
                f.write(f"**Test Case {i}:**\n")
                f.write(f"{display_text}\n")
                f.write("-" * 20 + "\n")
            f.write("\n")

if __name__ == "__main__":
    # Clear file first
    with open("test_cases.txt", "w", encoding="utf-8") as f:
        f.write("Test Cases Extracted\n")
    extract_test_cases()
