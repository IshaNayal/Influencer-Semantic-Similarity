import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from scipy.stats import ttest_rel

# Parameters
DATASET_PATH = 'final_results.csv'  # Update if needed
K = 5  # Number of nearest neighbors
MODEL_NAME = 'all-MiniLM-L6-v2'

# Load data
df = pd.read_csv(DATASET_PATH)

# Check required columns
df = df[['Influencer', 'NonSponsoredPost', 'OriginalSponsoredPost', 'GeneratedSponsoredPost']]

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Precompute embeddings for all posts
df['NonSponsoredPost_emb'] = list(model.encode(df['NonSponsoredPost'].tolist(), show_progress_bar=True))
df['OriginalSponsoredPost_emb'] = list(model.encode(df['OriginalSponsoredPost'].tolist(), show_progress_bar=True))
df['GeneratedSponsoredPost_emb'] = list(model.encode(df['GeneratedSponsoredPost'].tolist(), show_progress_bar=True))

# Group organic posts by influencer for reference tone space
influencer_to_organic_emb = df.groupby('Influencer')['NonSponsoredPost_emb'].apply(list).to_dict()

original_distances = []
generated_distances = []

for idx, row in df.iterrows():
    influencer = row['Influencer']
    organic_embs = np.array(influencer_to_organic_emb[influencer])

    # PN_distance_original
    orig_emb = np.array(row['OriginalSponsoredPost_emb']).reshape(1, -1)
    orig_distances = cdist(orig_emb, organic_embs, metric='cosine')[0]
    orig_knn = np.sort(orig_distances)[:K]
    pn_distance_original = np.mean(orig_knn)
    original_distances.append(pn_distance_original)

    # PN_distance_generated
    gen_emb = np.array(row['GeneratedSponsoredPost_emb']).reshape(1, -1)
    gen_distances = cdist(gen_emb, organic_embs, metric='cosine')[0]
    gen_knn = np.sort(gen_distances)[:K]
    pn_distance_generated = np.mean(gen_knn)
    generated_distances.append(pn_distance_generated)

# Paired t-test
original_distances = np.array(original_distances)
generated_distances = np.array(generated_distances)

t_stat, p_value = ttest_rel(original_distances, generated_distances)

# Summary statistics
mean_original = np.mean(original_distances)
mean_generated = np.mean(generated_distances)
num_posts = len(original_distances)

print(f"Number of posts analyzed: {num_posts}")
print(f"Mean Original Distance: {mean_original:.4f}")
print(f"Mean Generated Distance: {mean_generated:.4f}")
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.6f}")

# Interpretation
if mean_generated < mean_original and p_value < 0.05:
    print("\nResult:")
    print("AI-generated sponsored posts are significantly closer to unsponsored influencer content than original sponsored posts.")
else:
    print("\nResult:")
    print("No statistically significant evidence that AI-generated sponsored posts are closer to influencer tone than original sponsored posts.")
