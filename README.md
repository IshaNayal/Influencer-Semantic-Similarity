Influencer Style Rewriter & Semantic Similarity Analysis
========================================================

An end-to-end Machine Learning pipeline that rewrites **sponsored influencer posts into organic-style content** and evaluates how closely AI-generated posts match the influencer’s natural writing style using **semantic similarity and statistical testing**.

This project fine-tunes **per-influencer models**, generates rewrites, and provides **quantitative evidence** that AI-generated sponsored posts resemble organic posts better than original sponsored posts.

Project Features
----------------

*   Influencer-specific model training
    
*   Sponsored → Organic style rewriting
    
*   Semantic similarity analysis
    
*   Post-Neighbourhood (PN) distance evaluation
    
*   Statistical significance testing (t-test)
    
*   Per-influencer evaluation
    
*   Streamlit UI for testing rewrites
    
*   Reproducible ML pipeline
    

Project Structure
-----------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   .│├── comprehensive_pipeline.py├── semantic_similarity_analysis.py├── influencer_semantic_similarity.py├── analyze_semantic_similarity.py│├── streamlit_app.py│├── organic_data.csv├── training_data.csv├── top5_influencers.csv│├── final_results.csv├── final_comprehensive_report.csv├── similarity_stats.csv│└── README.md   `

Problem Statement
-----------------

Sponsored influencer posts often sound unnatural and different from the influencer’s organic content.

This project solves this problem by:

*   Learning influencer writing style
    
*   Rewriting sponsored posts
    
*   Measuring style similarity
    
*   Providing statistical validation
    

Workflow
--------

### 1\. Data Preparation

Datasets used:

### organic\_data.csv

Contains organic (non-sponsored) influencer posts.

### training\_data.csv

Contains sponsored posts used during early experiments.

### top5\_influencers.csv

Final dataset used for semantic analysis.

Columns:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   non_sponsored_textoriginal_sponsored_textmodified_sponsored_text   `

### 2\. Influencer Selection

The pipeline selects the **top 5 influencers** based on post count.

For each influencer:

*   Older posts → Training
    
*   Recent posts → Evaluation
    

Evaluation set:

*   Latest 100 posts
    
*   10–15 sponsored posts
    

Sponsored posts are identified using:

*   Dataset labels
    
*   Keyword detection (fallback)
    

Model Training
--------------

### Generator Model

Base model:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   GPT-2   `

Fine-tuning method:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   LoRA (Low Rank Adaptation)   `

Purpose:

Generate organic-style sponsored posts.

### Embedding Model

Base model:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   sentence-transformers/all-MiniLM-L6-v2   `

Training method:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   SimCSE-style contrastive learning   `

Purpose:

Learn influencer writing style.

Semantic Similarity Analysis
----------------------------

### Embedding Model

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   sentence-transformers/all-mpnet-base-v2   `

Used to generate embeddings for:

*   Organic posts
    
*   Original sponsored posts
    
*   Modified sponsored posts
    

Post-Neighbourhood (PN) Distance
--------------------------------

PN distance measures how close a sponsored post is to organic posts.

### Method

For each sponsored post:

1.  Compute cosine distance to all organic posts
    
2.  Select K nearest neighbors
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   K = 5   `

1.  Compute average distance
    

Two lists are created:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Original PN DistancesModified PN Distances   `

Statistical Testing
-------------------

A **paired t-test** compares:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Original PN DistancesvsModified PN Distances   `

Metrics computed:

*   Mean Original Distance
    
*   Mean Modified Distance
    
*   t-statistic
    
*   p-value
    

### Interpretation

If:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Mean Modified Distance < Mean Original Distance   `

Then:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   AI-generated sponsored posts match influencer tone better.   `

Scripts
-------

### comprehensive\_pipeline.py

End-to-end pipeline:

*   Load dataset
    
*   Train models
    
*   Generate rewrites
    
*   Compute distances
    
*   Save results
    

### semantic\_similarity\_analysis.py

Final production script:

*   PN distance computation
    
*   Statistical testing
    
*   CSV export
    

### influencer\_semantic\_similarity.py

Advanced analysis:

*   Influencer-wise evaluation
    
*   Debug outputs
    

### analyze\_semantic\_similarity.py

Initial prototype script.

Streamlit App
-------------

Interactive UI for testing rewrites.

### Features

*   Select influencer
    
*   Enter sponsored post
    
*   Generate rewrite
    
*   View similarity scores
    

Run:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit run streamlit_app.py   `

Installation
------------

Clone repository:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`git clone` 

Install dependencies:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

How to Run
----------

### Run Training Pipeline

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python comprehensive_pipeline.py   `

### Run Semantic Analysis

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python semantic_similarity_analysis.py   `

### Run Streamlit App

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit run streamlit_app.py   `

Output Files
------------

### final\_comprehensive\_report.csv

Influencer-level statistics:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   InfluencerSponsored CountAvg Original DistanceAvg Modified DistanceImprovement   `

### final\_results.csv

Post-level statistics:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   InfluencerSponsored PostModified PostOriginal DistanceModified DistanceImprovement   `

### similarity\_stats.csv

Statistical results:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ModelPairs_AnalyzedMean_Original_DistanceMean_Modified_Distancet_statisticp_valueInterpretation   `

Example Result
--------------

Sample experiment showed:

*   AI-generated sponsored posts had lower PN distances
    
*   Statistical tests showed significant improvement
    
*   Generated posts matched influencer tone better
    

Technologies Used
-----------------

*   Python
    
*   PyTorch
    
*   HuggingFace Transformers
    
*   SentenceTransformers
    
*   LoRA
    
*   Streamlit
    
*   Pandas
    
*   NumPy
    
*   SciPy
    
*   Scikit-learn
    

Limitations
-----------

*   Some influencers have few sponsored posts
    
*   Keyword detection may introduce noise
    
*   GPT-2 limits generation quality
    
*   Dataset size affects performance
    

Future Work
-----------

Possible improvements:

*   Larger datasets
    
*   Better sponsored detection
    
*   Stronger LLMs
    
*   RAG-based rewriting
    
*   Better embeddings
    

Author
------

**Isha Nayal**

