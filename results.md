# Semantic Similarity Analysis Results

## Summary of Conclusions

This project evaluated whether AI-generated (modified) sponsored content is semantically closer to an influencer’s organic (non-sponsored) posts than the original sponsored content. The analysis used sentence embeddings and a Post-Neighbourhood (PN) distance approach, followed by statistical testing.

### Key Findings
- **Mean Original Distance:** The average semantic distance between original sponsored posts and organic influencer posts.
- **Mean Modified Distance:** The average semantic distance between AI-generated sponsored posts and organic influencer posts.
- **t-statistic & p-value:** Results of a paired t-test comparing the two distance lists.
- **Interpretation:**
  - If the mean modified distance is lower than the mean original distance and the p-value is statistically significant, it means AI-generated sponsored content is closer to the influencer’s natural tone.

### Example Results (from sample run)
- **Number of posts analyzed:** 10
- **Mean Original Distance:** 0.4746
- **Mean Modified Distance:** 0.4396
- **t-statistic:** 11.2881
- **p-value:** 0.0000
- **Interpretation:** Modified sponsored content is closer to organic influencer tone

## Explanation of Scores

- **Cosine Distance:**
  - Ranges from 0 (identical) to 2 (opposite direction), but for normalized embeddings, practical range is 0 (identical) to 1 (completely dissimilar).
  - Lower values mean higher semantic similarity.

- **PN Distance (Post-Neighbourhood Distance):**
  - For each sponsored post, the average cosine distance to its 5 nearest organic posts is computed.
  - This measures how close a sponsored post is to the influencer’s typical (organic) tone.

- **Mean Distances:**
  - The mean of all PN distances for original and modified sponsored posts, respectively.
  - Lower mean indicates content is more similar to the influencer’s organic style.

- **t-statistic & p-value:**
  - The paired t-test checks if the difference between original and modified distances is statistically significant.
  - A low p-value (< 0.05) means the difference is unlikely due to chance.

## Conclusion

- In this analysis, AI-generated sponsored content was found to be significantly closer to the influencer’s organic tone than the original sponsored content.
- This suggests that AI can help brands create sponsored posts that better match the authentic voice of influencers, potentially improving engagement and audience trust.

---

For more details, see the full workflow and code in README.md and the raw scores in similarity_stats.csv.
