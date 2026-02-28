
# --- New Implementation for 3-way semantic similarity analysis ---
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_rel

# File paths
DATA_FILE = 'top5_influencers.csv'
RESULTS_FILE = 'similarity_stats.csv'
MODEL_NAME = 'all-mpnet-base-v2'
K = 5  # Number of nearest neighbors

# Load data
print(f"Loading data from {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

# Check columns
required_cols = ['non_sponsored_text', 'original_sponsored_text', 'modified_sponsored_text']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Prepare lists
organic_posts = df['non_sponsored_text'].tolist()
original_sponsored = df['original_sponsored_text'].tolist()
modified_sponsored = df['modified_sponsored_text'].tolist()

print(f"Number of posts analyzed: {len(original_sponsored)}")

# Generate embeddings
print(f"Loading SentenceTransformer model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

organic_embs = model.encode(organic_posts, show_progress_bar=True)
original_embs = model.encode(original_sponsored, show_progress_bar=True)
modified_embs = model.encode(modified_sponsored, show_progress_bar=True)

# Compute PN distances
original_pn_distances = []
modified_pn_distances = []

for i in range(len(original_sponsored)):
    # Cosine distances to all organic posts
    orig_dists = 1.0 - cosine_similarity([original_embs[i]], organic_embs)[0]
    mod_dists = 1.0 - cosine_similarity([modified_embs[i]], organic_embs)[0]
    # 5 nearest neighbors
    orig_knn = np.sort(orig_dists)[:K]
    mod_knn = np.sort(mod_dists)[:K]
    original_pn_distances.append(np.mean(orig_knn))
    modified_pn_distances.append(np.mean(mod_knn))

# Print first 5 PN distances for debug
print("First 5 Original PN Distances:", original_pn_distances[:5])
print("First 5 Modified PN Distances:", modified_pn_distances[:5])

# Paired t-test
mean_orig = np.mean(original_pn_distances)
mean_mod = np.mean(modified_pn_distances)
t_stat, p_value = ttest_rel(original_pn_distances, modified_pn_distances)

# Interpretation
if mean_mod < mean_orig:
    interpretation = "Modified sponsored content is closer to organic influencer tone"
else:
    interpretation = "Modified sponsored content is further from organic influencer tone"

# Save results
stats = pd.DataFrame({
    'Model': [MODEL_NAME],
    'Pairs_Analyzed': [len(original_pn_distances)],
    'Mean_Original_Distance': [mean_orig],
    'Mean_Modified_Distance': [mean_mod],
    't_statistic': [t_stat],
    'p_value': [p_value],
    'Interpretation': [interpretation]
})
stats.to_csv(RESULTS_FILE, index=False)

# Print summary
print("\n" + "="*50)
print("SEMANTIC SIMILARITY RESULTS")
print(f"Model used: {MODEL_NAME}")
print("="*50)
print(f"Number of posts analyzed: {len(original_pn_distances)}")
print(f"Mean Original Distance: {mean_orig:.4f}")
print(f"Mean Modified Distance: {mean_mod:.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Interpretation: {interpretation}")
print("="*50)
