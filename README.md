# Influencer Semantic Similarity Analysis

## Project Overview
This project analyzes the semantic similarity between influencer organic (non-sponsored) content, original sponsored posts, and AI-generated sponsored posts. The goal is to provide quantitative and statistical evidence that AI-generated sponsored content matches the influencer's natural tone better than the original sponsored content.

## Workflow Summary

### 1. Data Preparation
- **Datasets Used:**
   - `organic_data.csv`: Contains organic (non-sponsored) influencer posts.
   - `training_data.csv`: Contains sponsored posts (initially used for pairing, later replaced).
   - `top5_influencers.csv`: Final dataset with three columns:
      - `non_sponsored_text`: Organic influencer posts
      - `original_sponsored_text`: Original brand-sponsored posts
      - `modified_sponsored_text`: AI-generated sponsored posts

### 2. Script Development
- **Scripts Created:**
   - `analyze_semantic_similarity.py`: Initial script for basic similarity analysis.
   - `influencer_semantic_similarity.py`: Advanced script for influencer-wise PN distance and statistical testing.
   - `semantic_similarity_analysis.py`: Final, production-ready script for 3-way comparison using top5_influencers.csv.

### 3. Embedding Generation
- Used the `sentence-transformers` library with the `all-mpnet-base-v2` model to generate sentence embeddings for all posts.

### 4. Post-Neighbourhood (PN) Distance Calculation
- For each sponsored post (original and modified):
   - Calculated cosine distances to all organic posts.
   - Selected the 5 nearest neighbors (K=5).
   - Computed the average distance (PN distance).
- Built two lists:
   - `Original_PN_Distances`: For original sponsored posts
   - `Modified_PN_Distances`: For AI-generated sponsored posts

### 5. Statistical Testing
- Performed a paired t-test between the two PN distance lists.
- Calculated:
   - Mean Original Distance
   - Mean Modified Distance
   - t-statistic
   - p-value
- Interpretation:
   - If Mean Modified < Mean Original: "Modified sponsored content is closer to organic influencer tone"
   - Else: "Modified sponsored content is further from organic influencer tone"

### 6. Output and Results
- Results are printed in the terminal with summary statistics and interpretation.
- Results are saved to `similarity_stats.csv` with columns:
   - Model
   - Pairs_Analyzed
   - Mean_Original_Distance
   - Mean_Modified_Distance
   - t_statistic
   - p_value
   - Interpretation
- Debug prints include:
   - Dataset size
   - Number of pairs analyzed
   - First 5 PN distances for both original and modified posts

### 7. Key Findings (Sample Run)
- The script showed that AI-generated sponsored content is statistically significantly closer to organic influencer tone than original sponsored content (for the sample data).

## How to Use
1. Place your data in `top5_influencers.csv` with the required columns.
2. Run `semantic_similarity_analysis.py` in your Python environment.
3. View results in the terminal and in `similarity_stats.csv`.

## Requirements
- Python 3.7+
- sentence-transformers
- pandas
- numpy
- scipy
- scikit-learn

## Example Command
```bash
python semantic_similarity_analysis.py
```

## Project Status
- All scripts and analyses are complete and reproducible.
- You can add more data to `top5_influencers.csv` and rerun the script for updated results.

---

For any further customization or questions, please ask!
# Influencer Style Rewriter - Internship Report

## Overview
This project builds a full pipeline to rewrite sponsored influencer posts so they read like organic, authentic content. It fine-tunes per-influencer models, generates modified versions of sponsored posts, and evaluates how close the rewrites are to real organic posts using embedding distance and KNN.

## What Was Done
- Collected and loaded influencer post data from a Hugging Face dataset (with local CSV fallback).
- Selected the top 5 influencers by post count.
- Split each influencer's posts into a training set and a recent evaluation set.
- Fine-tuned a text generation model (GPT-2 + LoRA) per influencer.
- Fine-tuned an embedding model (SentenceTransformers) per influencer using contrastive learning.
- Generated organic-style rewrites for sponsored posts.
- Measured similarity using KNN distances to the 5 nearest non-sponsored posts.
- Built a Streamlit UI to test rewrites and view results.

## Pipeline Summary (Step by Step)
1. Load full dataset from Hugging Face: ishanayal16/influencer_data.
2. For each of the top 5 influencers:
   - Exclude the latest 100 posts for evaluation.
   - Use the remaining posts to fine-tune models.
   - From the latest 100 posts, extract 10-15 sponsored posts (label-based or keyword-based).
3. Fine-tune models:
   - Generator: GPT-2 with LoRA adapters.
   - Embeddings: sentence-transformers/all-MiniLM-L6-v2 with SimCSE-style contrastive learning.
4. Generate rewrites for the sponsored posts.
5. Compute KNN distances to 5 nearest non-sponsored posts.
6. Save results to CSV for reporting.

## Streamlit UI
The app allows:
- Selecting an influencer.
- Pasting a sponsored post for rewriting.
- Generating an organic-style version.
- Viewing similarity scores.

File: streamlit_app.py

## Key Scripts and Files
- comprehensive_pipeline.py: End-to-end training + evaluation pipeline.
- streamlit_app.py: Streamlit UI for testing rewrites.
- extract_test_cases.py: Creates sample sponsored test cases.
- get_deepika_posts.py: Pulls Deepika-specific samples.
- final_comprehensive_report.csv: Summary results per influencer.
- final_results.csv: Per-post results.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the pipeline:
   python comprehensive_pipeline.py

3. Start the Streamlit app:
   streamlit run streamlit_app.py

## Outputs
- final_comprehensive_report.csv
  Columns: Influencer, Sponsored Count, Avg Original Dist, Avg Modified Dist, Improvement
- final_results.csv
  Per-post original vs modified distances and improvements.

## Notes and Observations
- The pipeline is fully implemented and runs end-to-end.
- If sponsored labels are missing, keyword detection is used.
- The evaluation uses cosine distance with KNN (k=5).

## Limitations and Future Work
- Some influencers have few explicit sponsored posts; sampling is used for consistency.
- Generation quality depends on the base model and available training data.
- Future improvements could include better sponsored detection and stronger generation models.

## Author
Internship Project - Influencer Style Rewriter
