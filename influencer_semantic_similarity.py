import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_rel
import time
import os

def get_embeddings(texts, model_name='sentence-transformers/all-mpnet-base-v2'):
    """
    Generates embeddings for a list of texts using SentenceTransformer.
    Uses 'all-mpnet-base-v2' as requested for better writing style capture.
    
    Time Complexity: O(N * L), where N is number of texts and L is average length.
    """
    print(f"Generating embeddings for {len(texts)} texts using {model_name}...")
    start_time = time.time()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Done in {time.time() - start_time:.2f} seconds.")
    return embeddings

def compute_pn_distance_influencer_wise(influencer_list, s_texts, m_texts, ns_df, k=10):
    """
    Computes PN-distance influencer-wise. 
    Each sponsored post is only compared to non-sponsored posts from the same influencer.
    
    PN_distance = mean(nearest_k_distances) where distance = 1 - cosine_similarity.
    
    Time Complexity: O(I * (N_i * Q_i)), where I is influencers, 
    N_i is non-sponsored posts per influencer, Q_i is queries per influencer.
    """
    print("Loading model...")
    model_used = "all-mpnet-base-v2"
    try:
        # Try MPNet first as requested
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    except Exception as e:
        print(f"Error loading MPNet: {e}")
        print("Falling back to all-MiniLM-L6-v2...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        model_used = "all-MiniLM-L6-v2"
    
    original_distances = []
    generated_distances = []
    unique_influencers = set(influencer_list)
    # Pre-filter NS posts to remove sponsored ones
    keywords = ['#ad', '#sponsored', 'sponsored', ' ad ', 'promo', 'partner', 'gifted', 'collab']
    pattern = '|'.join(keywords)
    ns_df = ns_df[~ns_df['caption'].str.contains(pattern, case=False, na=False)].copy()
    k_nn = 5
    for influencer in unique_influencers:
        print(f"Processing influencer: {influencer}")
        ns_texts_inf = ns_df[ns_df['Name_x'] == influencer]['caption'].tolist()
        if len(ns_texts_inf) < k_nn:
            print(f"  Skipping {influencer}: Not enough non-sponsored posts (found {len(ns_texts_inf)}, need {k_nn})")
            continue
        indices = [i for i, x in enumerate(influencer_list) if x == influencer]
        s_inf = [s_texts[i] for i in indices]
        m_inf = [m_texts[i] for i in indices]
        ns_emb = model.encode(ns_texts_inf, show_progress_bar=False)
        s_emb = model.encode(s_inf, show_progress_bar=False)
        m_emb = model.encode(m_inf, show_progress_bar=False)
        # For each sponsored post, compute cosine distances to all NS posts
        for s_vec in s_emb:
            dists = 1.0 - cosine_similarity([s_vec], ns_emb)[0]
            topk = np.sort(dists)[:k_nn]
            original_distances.append(np.mean(topk))
        for m_vec in m_emb:
            dists = 1.0 - cosine_similarity([m_vec], ns_emb)[0]
            topk = np.sort(dists)[:k_nn]
            generated_distances.append(np.mean(topk))
    return np.array(original_distances), np.array(generated_distances), model_used

def run_paired_ttest(original_distances, modified_distances):
    """
    Performs a paired t-test using scipy.stats.ttest_rel.
    """
    t_stat, p_value = ttest_rel(original_distances, modified_distances)
    return t_stat, p_value

def main():
    # 1. Load Data
    results_file = "final_results.csv"
    organic_file = "organic_data.csv"
    
    if not os.path.exists(results_file) or not os.path.exists(organic_file):
        print("Required CSV files missing. Please ensure final_results.csv and organic_data.csv exist.")
        return
        
    df_results = pd.read_csv(results_file)
    df_organic = pd.read_csv(organic_file)

    # Select top 100 organic posts per influencer by engagement
    df_organic['engagement'] = df_organic['like_count'] + df_organic['comment_count']
    top_organic = (
        df_organic.sort_values(['Name_x', 'engagement'], ascending=[True, False])
        .groupby('Name_x')
        .head(100)
        .reset_index(drop=True)
    )

    # Filter results to top 100 per influencer if more than 100
    top_results = (
        df_results.groupby('Influencer')
        .head(100)
        .reset_index(drop=True)
    )

    influencers = top_results['Influencer'].tolist()
    s_texts = top_results['Original Sponsored'].tolist()
    m_texts = top_results['Modified'].tolist()

    print("Starting influencer-wise PN-distance calculation (top 100 per influencer)...")
    orig_dist, mod_dist, model_name = compute_pn_distance_influencer_wise(
        influencers, s_texts, m_texts, top_organic, k=10
    )
    
    if len(orig_dist) == 0:
        print("No data pairs processed. Check if influencers match in both datasets.")
        return

    # 3. Perform Statistical Test
    t_stat, p_value = run_paired_ttest(orig_dist, mod_dist)
    
    mean_orig = np.mean(orig_dist)
    mean_mod = np.mean(mod_dist)
    
    # 4. Print and Save Results
    print("\n" + "="*50)
    print("INFLUENCER-WISE SEMANTIC SIMILARITY RESULTS")
    print(f"Model used: {model_name}")
    print("="*50)
    print(f"Number of pairs analyzed: {len(orig_dist)}")
    print(f"Mean Original Distance:  {mean_orig:.4f}")
    print(f"Mean Modified Distance:  {mean_mod:.4f}")
    print(f"t-statistic:             {t_stat:.4f}")
    print(f"p-value:                 {p_value:.6e}")
    print("-" * 50)

    interpretation = (
        "Modified sponsored content is significantly closer to non-sponsored content"
        if mean_mod < mean_orig else
        "Modified sponsored content is significantly further from non-sponsored content"
    )
    print(f"RESULT: {interpretation}")
    print("="*50 + "\n")

    # Save to CSV
    stats = pd.DataFrame({
        'Model': [model_name],
        'Pairs_Analyzed': [len(orig_dist)],
        'Mean_Original_Distance': [mean_orig],
        'Mean_Modified_Distance': [mean_mod],
        't_statistic': [t_stat],
        'p_value': [p_value],
        'Interpretation': [interpretation]
    })
    stats.to_csv('similarity_stats.csv', index=False)

if __name__ == "__main__":
    main()
