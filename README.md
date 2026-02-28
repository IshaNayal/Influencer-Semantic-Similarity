#Influencer Style Rewriter & Semantic Similarity Analysis

Internship Project Report
Author: Isha Nayal

1. Project Overview

This project develops a complete pipeline to rewrite sponsored influencer posts so that they resemble organic content while preserving the original message.

The system uses fine-tuned language models and embedding models to generate organic-style sponsored posts and then quantitatively evaluates how closely the generated posts match the influencer's natural writing style.

The primary goal is to provide statistical evidence that AI-generated sponsored content matches influencer tone better than original sponsored posts.

2. Objectives

The main objectives of the project were:

Rewrite sponsored posts into organic-style influencer content

Train per-influencer AI models

Measure similarity between:

Organic posts

Original sponsored posts

AI-generated sponsored posts

Provide quantitative evaluation

Build a reproducible ML pipeline

Develop an interactive UI for testing

3. Dataset Preparation
Datasets Used
1. organic_data.csv

Contains organic (non-sponsored) influencer posts.

2. training_data.csv

Contains sponsored posts used during early experimentation.

3. top5_influencers.csv (Final Dataset)

Final dataset used for semantic analysis.

Columns:

non_sponsored_text
Organic influencer posts

original_sponsored_text
Original brand-sponsored posts

modified_sponsored_text
AI-generated sponsored posts

4. System Workflow
Step 1 — Data Loading

The dataset is loaded from:

Hugging Face dataset:

ishanayal16/influencer_data

Local CSV fallback if online loading fails.

Step 2 — Influencer Selection

The pipeline selects the top 5 influencers based on post count.

For each influencer:

Older posts → Training

Recent posts → Evaluation

Evaluation set:

Latest 100 posts

10–15 sponsored posts selected

Sponsored posts are identified using:

Labels (if available)

Keyword detection (fallback)

Step 3 — Model Training

Two models were fine-tuned per influencer.

3.1 Generator Model

Base Model:

GPT-2

Method:

LoRA fine-tuning

Purpose:

Generate organic-style rewrites of sponsored posts.

3.2 Embedding Model

Base Model:

SentenceTransformers

Model:

all-MiniLM-L6-v2

Training Method:

SimCSE-style contrastive learning

Purpose:

Generate embeddings that better capture influencer style.

5. Semantic Similarity Analysis
Embedding Model

Sentence embeddings were generated using:

all-mpnet-base-v2

These embeddings were used to compare:

Organic posts

Original sponsored posts

Modified sponsored posts

6. Post-Neighbourhood (PN) Distance

To evaluate style similarity, Post-Neighbourhood (PN) Distance was used.

Method

For each sponsored post:

Compute cosine distance to all organic posts.

Select the K nearest neighbors (K = 5).

Compute the average distance.

Two distance sets were generated:

Original PN Distances

Modified PN Distances

7. Statistical Testing

A paired t-test was performed to compare:

Original PN distances

Modified PN distances

Metrics calculated:

Mean Original Distance

Mean Modified Distance

t-statistic

p-value

Interpretation Rule

If:

Mean Modified Distance < Mean Original Distance

Then:

AI-generated sponsored content is closer to organic influencer tone.

Otherwise:

Original sponsored content is closer to organic influencer tone.

8. Scripts Developed
1. analyze_semantic_similarity.py

Initial prototype for similarity analysis.

Features:

Basic cosine similarity

Preliminary testing

2. influencer_semantic_similarity.py

Advanced analysis.

Features:

Influencer-wise PN distance

Statistical testing

Debug outputs

3. semantic_similarity_analysis.py

Final production script.

Features:

3-way comparison

Statistical testing

CSV export

Reproducible results

9. Output Files
similarity_stats.csv

Columns:

Model

Pairs_Analyzed

Mean_Original_Distance

Mean_Modified_Distance

t_statistic

p_value

Interpretation

final_comprehensive_report.csv

Contains influencer-level statistics:

Influencer

Sponsored Count

Avg Original Distance

Avg Modified Distance

Improvement

final_results.csv

Contains post-level results:

Influencer

Sponsored Post

Modified Post

Original Distance

Modified Distance

Improvement

10. Streamlit Interface

A Streamlit application was developed to interactively test the model.

Features

Select an influencer

Enter a sponsored post

Generate organic-style rewrite

View similarity scores

File:

streamlit_app.py
11. Pipeline Execution
Install Dependencies
pip install -r requirements.txt
Run Training Pipeline
python comprehensive_pipeline.py
Run Semantic Analysis
python semantic_similarity_analysis.py
Start Streamlit App
streamlit run streamlit_app.py
12. Example Results

A sample run showed:

AI-generated sponsored posts had lower PN distances

Paired t-test showed statistically significant improvement

Generated posts matched influencer tone better than original sponsored posts.

13. Key Contributions

This project provides:

✔ End-to-end ML pipeline
✔ Per-influencer fine-tuning
✔ Semantic similarity evaluation
✔ Statistical validation
✔ Interactive UI
✔ Reproducible experiments

14. Limitations

Some influencers have limited sponsored posts.

Keyword-based sponsored detection may introduce noise.

GPT-2 limits generation quality.

Dataset size affects model performance.

15. Future Improvements

Possible improvements include:

Better sponsored-post detection

Larger datasets

Stronger LLMs

RAG-based generation

Style-aware embeddings

16. Project Status

The project is fully implemented and reproducible.

New data can be added to:

top5_influencers.csv

and the pipeline can be rerun to update results.

17. Technologies Used

Python

PyTorch

Hugging Face Transformers

SentenceTransformers

LoRA

Streamlit

Pandas

NumPy

SciPy

Scikit-learn
