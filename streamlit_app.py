import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import pandas as pd
import requests

# Page Config
st.set_page_config(page_title="Influencer Style Rewriter", layout="wide")

# Constants
BASE_MODEL = "gpt2"
RESULTS_DIR = "comprehensive_results"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"
SUMMARY_CSV = "final_comprehensive_report.csv"
DETAILS_CSV = "final_results.csv"

INFLUENCER_CONFIG = {
    'thesuburbansoapbox': {'niche': 'Food & Recipes', 'icon': '🍳'},
    'deepika.padukone.the.princess': {'niche': 'Celebrity & Fashion', 'icon': '👗'},
    'twopeasinablog': {'niche': 'Lifestyle & Fashion', 'icon': '👯‍♀️'},
    'absolutelyairs': {'niche': 'Air Fryer Recipes', 'icon': '🍗'},
    'soheatherblog': {'niche': 'Fashion & Style', 'icon': '👠'}
}

@st.cache_resource
def load_data():
    """Load the dataset once and cache it."""
    try:
        # Try loading from Hugging Face
        hf_dataset = load_dataset("ishanayal16/influencer_data")
        df = hf_dataset['train'].to_pandas()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Fallback to local if exists
        if os.path.exists("organic_data.csv"):
            return pd.read_csv("organic_data.csv")
        return pd.DataFrame()

@st.cache_resource
def load_embedding_model(influencer_name):
    """Load the fine-tuned embedding model or fallback to base."""
    # Check for fine-tuned model
    model_path = os.path.join(RESULTS_DIR, f"{influencer_name}_emb")
    if os.path.exists(model_path):
        try:
            return SentenceTransformer(model_path)
        except:
            pass
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_data
def load_results_tables():
    """Load summary and detailed results tables if present."""
    summary_df = pd.read_csv(SUMMARY_CSV) if os.path.exists(SUMMARY_CSV) else pd.DataFrame()
    details_df = pd.read_csv(DETAILS_CSV) if os.path.exists(DETAILS_CSV) else pd.DataFrame()
    return summary_df, details_df

@st.cache_resource
def load_model(influencer_name):
    """Load the fine-tuned LoRA model for the specific influencer."""
    model_path = os.path.join(RESULTS_DIR, f"{influencer_name}_gen")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model for {influencer_name}: {e}")
        return None, None

def generate_rewrite(model, tokenizer, text):
    """Generate organic rewrite using seed strategy."""
    # Seed Strategy: Take the first sentence (or part of it) to ground the topic
    sentences = re.split(r'[.!?\n]', text)
    seed = sentences[0]
    
    # If seed is too short, add the second sentence
    if len(seed) < 20 and len(sentences) > 1:
        seed += ". " + sentences[1]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(seed, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.8, # Slightly higher for creativity
            repetition_penalty=1.2, # Reduced from 1.5 to allow natural flow
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            top_k=50,
            no_repeat_ngram_size=3 # Increased from 2 to avoid blocking common phrases
        )
    
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if not gen_text.strip().endswith(('.', '!', '?')):
        gen_text = gen_text.rsplit('.', 1)[0] + "."
        
    return gen_text

def generate_with_ollama(text, influencer_name, niche, style_examples=None, temperature=0.7):
    """Generate rewrite using local Ollama (Llama 3.1) with style matching rules."""
    
    organic_posts_str = ""
    if style_examples:
        organic_posts_str = "\n---\n".join(style_examples)
    
    prompt = f"""
You are an expert social media manager working on the research project: 
"Influencer Engagement Prediction and Creator-Consistent Rewriting of Sponsored Content"

Goal: Rewrite the given sponsored caption so that it strongly matches the influencer's writing style and reduces semantic distance to organic posts.

-----------------------------------------
STYLE EXAMPLES (Organic Posts from {influencer_name})
-----------------------------------------
{organic_posts_str}

-----------------------------------------
SPONSORED CAPTION TO REWRITE
-----------------------------------------
{text}

-----------------------------------------
STRICT REWRITING RULES
-----------------------------------------
1. Match influencer tone exactly
2. Match sentence length pattern
3. Match emoji usage
4. Match punctuation style
5. Match storytelling style
6. Use first-person voice if influencer uses it
7. Keep product name unchanged
8. Keep claims unchanged
9. Avoid marketing language (no "Introducing", "Experience", "Discover")
10. Avoid corporate tone / advertisement tone
11. Avoid "Buy now" type sentences
12. Make it sound like a personal experience

TASK:
Generate ONE rewritten caption.
The rewrite must be more natural, personal, and similar to organic captions.
Return ONLY the rewritten text, no other commentary.
"""
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"Error from Ollama: {response.status_code}"
    except Exception as e:
        return f"Ollama Connection Error: {e}. Ensure Ollama is running."

def generate_best_with_ollama(text, influencer_name, niche, emb_model, df, candidate_count=3):
    """Generate multiple candidates and pick the closest style match."""
    # Get organic examples for the prompt
    if 'Name_x' in df.columns:
        subset = df[df['Name_x'] == influencer_name]
        # Filter organic
        keywords = ['#ad', '#sponsored', '#partner', '#gifted', '#collab']
        pattern = '|'.join(keywords)
        organic_subset = subset[~subset['caption'].str.contains(pattern, case=False, na=False)]
        
        if len(organic_subset) < 5:
            style_examples = subset['caption'].sample(min(len(subset), 10)).tolist()
        else:
            style_examples = organic_subset['caption'].sample(min(len(organic_subset), 10)).tolist()
    else:
        style_examples = []

    temperatures = [0.6, 0.7, 0.85, 1.0, 1.1]
    temps = temperatures[:max(1, min(candidate_count, len(temperatures)))]

    best_text = None
    best_avg = 0.0
    best_max = 0.0

    for temp in temps:
        candidate = generate_with_ollama(text, influencer_name, niche, style_examples=style_examples, temperature=temp)
        if candidate.startswith("Error from Ollama") or candidate.startswith("Ollama Connection Error"):
            continue

        avg_sim, max_sim = calculate_similarity_metrics(emb_model, candidate, influencer_name, df)
        if max_sim > best_max:
            best_text = candidate
            best_avg = avg_sim
            best_max = max_sim

    if best_text is None:
        return "Ollama is unavailable. Please start Ollama and try again.", 0.0, 0.0

    return best_text, best_avg, best_max

def calculate_similarity_metrics(emb_model, generated_text, influencer_name, df):
    """Calculate similarity between generated text and influencer's organic posts."""
    # Get random organic samples for this influencer
    if 'Name_x' in df.columns:
        subset = df[df['Name_x'] == influencer_name]
    else:
        return 0.0, 0.0 # Fail safe

    # Filter out potential sponsored posts if possible (simple keyword check)
    keywords = ['#ad', '#sponsored', '#partner', '#gifted', '#collab']
    pattern = '|'.join(keywords)
    organic_subset = subset[~subset['caption'].str.contains(pattern, case=False, na=False)]
    
    if len(organic_subset) < 5:
        samples = subset['caption'].sample(min(len(subset), 10)).tolist()
    else:
        samples = organic_subset['caption'].sample(min(len(organic_subset), 10)).tolist()
        
    # Embeddings
    gen_emb = emb_model.encode([generated_text])
    sample_embs = emb_model.encode(samples)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(gen_emb, sample_embs)[0]
    
    # Average and Max similarity
    avg_sim = np.mean(similarities)
    max_sim = np.max(similarities)
    
    return avg_sim, max_sim

# UI Layout
st.title("✨ Influencer Style Rewriter")
st.markdown("Convert **Sponsored/Promotional** posts into **Organic/Authentic** content that matches a specific influencer's style.")

# Results Overview
summary_df, details_df = load_results_tables()

# Check Ollama Status
def is_ollama_available():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

ollama_available = is_ollama_available()

# Sidebar
st.sidebar.header("Configuration")
selected_name = st.sidebar.selectbox("Select Influencer", list(INFLUENCER_CONFIG.keys()))
influencer_info = INFLUENCER_CONFIG[selected_name]

# Display Niche Info
st.sidebar.info(f"**Niche:** {influencer_info['niche']} {influencer_info['icon']}")
st.sidebar.warning("Note: Models are fine-tuned on specific niches. Inputting 'Fashion' text into a 'Food' influencer model may cause hallucinations (e.g., describing clothes as 'delicious').")

use_ollama = st.sidebar.checkbox("🚀 Use Llama 3.1 (Smarter Context)", value=ollama_available, disabled=not ollama_available, help="Uses local Ollama to better understand context and prevent hallucinations across niches.")
if not ollama_available:
    st.sidebar.caption("❌ Ollama not detected. Using basic GPT-2 model.")
candidate_count = st.sidebar.slider("Ollama candidates", min_value=1, max_value=5, value=3, help="Generate multiple rewrites and pick the closest style match.")

# Load Model
with st.spinner(f"Loading model for {selected_name}..."):
    model, tokenizer = load_model(selected_name)
    emb_model = load_embedding_model(selected_name)
    df = load_data()

def get_match_percentage(score):
    """Convert cosine similarity to a human-readable match percentage."""
    # Heuristic: 0.2 is random, 0.8 is perfect. Map 0.2-0.8 to 0-100%
    if score <= 0.2:
        return 0
    if score >= 0.8:
        return 100
    return int((score - 0.2) / 0.6 * 100)

if model:
    st.success(f"Model loaded for **{selected_name}**")

    # Input
    st.subheader("Input Sponsored Post")
    default_text = "I absolutely love this new summer collection from @FashionNova! #ad The fabric is so breathable and perfect for beach days. Use my code SUMMER20 for 20% off your entire order! #FashionNovaPartner"
    input_text = st.text_area("Paste text here:", value=default_text, height=150)

    if st.button("🔄 Rewrite as Organic"):
        if not input_text:
            st.warning("Please enter some text.")
        else:
            with st.spinner("Generating rewrite..."):
                if use_ollama:
                    rewritten_text, avg_sim, max_sim = generate_best_with_ollama(
                        input_text,
                        selected_name,
                        influencer_info['niche'],
                        emb_model,
                        df,
                        candidate_count=candidate_count,
                    )
                else:
                    rewritten_text = generate_rewrite(model, tokenizer, input_text)
                    # Calculate Similarity
                    avg_sim, max_sim = calculate_similarity_metrics(emb_model, rewritten_text, selected_name, df)
            
            # Display Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📢 Original (Sponsored)")
                st.info(input_text)
                
            with col2:
                st.markdown(f"### 🌿 {selected_name}'s Style (Organic)")
                st.success(rewritten_text)
                
                # Similarity Metric Display
                st.markdown("#### 📊 Style Similarity Score")
                
                # Interpret Score
                match_pct = get_match_percentage(max_sim)
                
                st.progress(match_pct / 100)
                st.caption(f"**{match_pct}% Match Confidence** (Based on semantic similarity to organic posts)")
                
                score_col1, score_col2 = st.columns(2)
                with score_col1:
                    st.metric("Avg Similarity", f"{avg_sim:.2f}")
                with score_col2:
                    st.metric("Max Similarity", f"{max_sim:.2f}")
                
                if max_sim > 0.5:
                    st.caption("✅ **High Similarity:** The generated text strongly matches the influencer's usual tone.")
                elif max_sim > 0.3:
                    st.caption("⚠️ **Moderate Similarity:** The text is related but might differ slightly in tone.")
                else:
                    st.caption("❌ **Low Similarity:** The text may feel off-brand or generic.")

            st.markdown("---")
            st.markdown("**Why this works:**")
            st.markdown("- **Topic Retention:** Uses the first sentence as a seed to keep the context.")
            st.markdown("- **Style Matching:** Uses a LoRA adapter fine-tuned on the influencer's non-sponsored posts.")
            st.markdown("- **Promo Removal:** The model naturally drifts away from hard-sell language because it wasn't trained on it.")
else:
    st.warning("Model not found. Please run the training pipeline first.")
