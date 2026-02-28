import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

# --- STEP 1: Load Data ---
# Load final results (sponsored original & generated)
results_df = pd.read_csv('final_results.csv')

# Load organic (unsponsored) posts
organic_df = pd.read_csv('organic_data.csv')

# Get top 5 influencers
top5 = results_df['Influencer'].value_counts().head(5).index.tolist()

# Filter for top 5 influencers only
results_df = results_df[results_df['Influencer'].isin(top5)]
organic_df = organic_df[organic_df['Name_x'].isin(top5)]

# --- STEP 2: Generate Embeddings ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute unsponsored embeddings for each influencer
unsponsored_embs = {}
for name in top5:
    posts = organic_df[organic_df['Name_x'] == name]['caption'].dropna().tolist()
    if posts:
        unsponsored_embs[name] = model.encode(posts, show_progress_bar=False)
    else:
        unsponsored_embs[name] = np.array([])

# --- STEP 3: Define PN_distance function ---
def pn_distance(post, influencer, unsponsored_embs, model, top_k=5):
    if len(unsponsored_embs[influencer]) == 0:
        return np.nan
    emb = model.encode([post])[0]
    sims = cosine_similarity([emb], unsponsored_embs[influencer])[0]
    topk_idx = np.argsort(sims)[-top_k:][::-1]
    topk_sims = sims[topk_idx]
    avg_dist = 1 - np.mean(topk_sims)
    return avg_dist

# --- STEP 4: Compute PN_distances for all pairs ---
pn_dist_original = []
pn_dist_generated = []

for idx, row in results_df.iterrows():
    influencer = row['Influencer']
    orig = row['Original Sponsored']
    gen = row['Modified']
    pn_dist_original.append(pn_distance(orig, influencer, unsponsored_embs, model))
    pn_dist_generated.append(pn_distance(gen, influencer, unsponsored_embs, model))

# Remove any NaNs (if any influencer had no unsponsored posts)
pn_dist_original = np.array(pn_dist_original)
pn_dist_generated = np.array(pn_dist_generated)
mask = ~np.isnan(pn_dist_original) & ~np.isnan(pn_dist_generated)
pn_dist_original = pn_dist_original[mask]
pn_dist_generated = pn_dist_generated[mask]

# --- STEP 5: Paired t-test ---
t_stat, p_value = ttest_rel(pn_dist_original, pn_dist_generated)
print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print("Generated content significantly reduces neighbourhood distance.")
else:
    print("No significant reduction in neighbourhood distance.")

# --- STEP 6: Visualization ---
plt.figure(figsize=(8,5))
data = pd.DataFrame({
    'PN_distance_original': pn_dist_original,
    'PN_distance_generated': pn_dist_generated
})
data_melt = data.melt(var_name='Type', value_name='PN_distance')
sns.boxplot(x='Type', y='PN_distance', data=data_melt)
plt.title('PN Distance Comparison (Original vs Generated)')
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(pn_dist_original, color='blue', label='Original', kde=True, stat='density', alpha=0.5)
sns.histplot(pn_dist_generated, color='green', label='Generated', kde=True, stat='density', alpha=0.5)
plt.legend()
plt.title('Histogram of PN Distances')
plt.show()

# Mean comparison
print(f"Mean PN_distance_original: {np.mean(pn_dist_original):.3f}")
print(f"Mean PN_distance_generated: {np.mean(pn_dist_generated):.3f}")

# --- STEP 7: Effect Size (Cohen's d) ---
def cohens_d(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

d = cohens_d(pn_dist_original, pn_dist_generated)
print(f"Cohen's d: {d:.3f}")
