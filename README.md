# Influencer Style Rewriter & Semantic Similarity Analysis

An end-to-end Machine Learning pipeline that rewrites sponsored influencer posts into organic-style content and evaluates how closely AI-generated posts match the influencer's natural writing style using semantic similarity and statistical testing.

This project fine-tunes per-influencer models, generates rewrites, and provides quantitative evidence that AI-generated sponsored posts resemble organic posts better than original sponsored posts.

---

## Project Features

- Influencer-specific model training
- Sponsored → Organic style rewriting
- Semantic similarity analysis
- Post-Neighbourhood (PN) distance evaluation
- Statistical significance testing (t-test)
- Per-influencer evaluation
- Streamlit UI for testing rewrites
- Reproducible ML pipeline

---

---

## Problem Statement

Sponsored influencer posts often sound unnatural and different from the influencer's organic content.

This project solves this problem by:

- Learning influencer writing style
- Rewriting sponsored posts
- Measuring style similarity
- Providing statistical validation

---

## Workflow

### 1. Data Preparation

### Datasets Used

#### organic_data.csv
Contains organic (non-sponsored) influencer posts.

#### training_data.csv
Contains sponsored posts used during early experiments.

#### top5_influencers.csv

Final dataset used for semantic analysis.

**Columns:**


---

### 2. Influencer Selection

The pipeline selects the **top 5 influencers** based on post count.

For each influencer:

- Older posts → Training
- Recent posts → Evaluation

Evaluation set:

- Latest 100 posts
- 10–15 sponsored posts

Sponsored posts are identified using:

- Dataset labels
- Keyword detection (fallback)

---

## Model Training

### Generator Model

**Base model:**

GPT-2
**Fine-tuning method:**

LoRA (Low Rank Adaptation)


**Purpose:**

Generate organic-style sponsored posts.

---

### Embedding Model

**Base model:**
sentence-transformers/all-MiniLM-L6-v2


**Training method:**
SimCSE-style contrastive learning


**Purpose:**

Learn influencer writing style.

---

## Semantic Similarity Analysis

### Embedding Model
sentence-transformers/all-mpnet-base-v2

Used to generate embeddings for:

- Organic posts
- Original sponsored posts
- Modified sponsored posts

---

## Post-Neighbourhood (PN) Distance

PN distance measures how close a sponsored post is to organic posts.

### Method

For each sponsored post:

1. Compute cosine distance to all organic posts
2. Select K nearest neighbors
K = 5
3. Compute average distance

Two lists are created:
Original PN Distances
Modified PN Distances


---

## Statistical Testing

A **paired t-test** compares:
Original PN Distances
vs
Modified PN Distances

### Metrics Computed

- Mean Original Distance
- Mean Modified Distance
- t-statistic
- p-value

### Interpretation

If: Mean Modified Distance < Mean Original Distance
Then:


AI-generated sponsored posts match influencer tone better.


---

## Scripts

### comprehensive_pipeline.py

End-to-end pipeline:

- Load dataset
- Train models
- Generate rewrites
- Compute distances
- Save results

---

### semantic_similarity_analysis.py

Final production script:

- PN distance computation
- Statistical testing
- CSV export

---

### influencer_semantic_similarity.py

Advanced analysis:

- Influencer-wise evaluation
- Debug outputs

---

### analyze_semantic_similarity.py

Initial prototype script.

---

## Streamlit App

Interactive UI for testing rewrites.

### Features

- Select influencer
- Enter sponsored post
- Generate rewrite
- View similarity scores

### Run


streamlit run streamlit_app.py


---

## Installation

### Clone Repository


git clone <your-repo-link>


### Install Dependencies


pip install -r requirements.txt


---

## How to Run

### Run Training Pipeline


python comprehensive_pipeline.py


### Run Semantic Analysis


python semantic_similarity_analysis.py


### Run Streamlit App


streamlit run streamlit_app.py


---

## Output Files

### final_comprehensive_report.csv

Influencer-level statistics:


Influencer
Sponsored Count
Avg Original Distance
Avg Modified Distance
Improvement


---

### final_results.csv

Post-level statistics:


Influencer
Sponsored Post
Modified Post
Original Distance
Modified Distance
Improvement


---

### similarity_stats.csv

Statistical results:


Model
Pairs_Analyzed
Mean_Original_Distance
Mean_Modified_Distance
t_statistic
p_value
Interpretation


---

## Example Result

Sample experiment showed:

- AI-generated sponsored posts had lower PN distances
- Statistical tests showed significant improvement
- Generated posts matched influencer tone better

---

## Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- SentenceTransformers
- LoRA
- Streamlit
- Pandas
- NumPy
- SciPy
- Scikit-learn

---

## Limitations

- Some influencers have few sponsored posts
- Keyword detection may introduce noise
- GPT-2 limits generation quality
- Dataset size affects performance

---

## Future Work

Possible improvements:

- Larger datasets
- Better sponsored detection
- Stronger LLMs
- RAG-based rewriting
- Better embeddings

---
**Isha Nayal**  
