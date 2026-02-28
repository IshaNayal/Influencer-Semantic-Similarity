import pandas as pd
import numpy as np
import os
import torch
import re
from collections.abc import Iterator
from sklearn.metrics.pairwise import cosine_similarity

# Hugging Face
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, load_dataset
import requests

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Constants
# Will be determined dynamically
OUTPUT_ROOT = "comprehensive_results"
MODEL_NAME_GEN = "gpt2"
MODEL_NAME_EMB = "sentence-transformers/all-mpnet-base-v2"
SPONSORED_MIN = 10
SPONSORED_MAX = 15
KNN_K = 5
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"

class ComprehensivePipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load Data from Hugging Face
        print("Loading dataset from Hugging Face: ishanayal16/influencer_data...")
        try:
            hf_dataset = load_dataset("ishanayal16/influencer_data", streaming=False)
            self.df_full = self._ensure_dataframe(hf_dataset["train"].to_pandas())
            print(f"Loaded {len(self.df_full)} total posts.")
        except Exception as e:
            print(f"Error loading HF dataset: {e}")
            # Fallback to local
            if os.path.exists("organic_data.csv"):
                print("Falling back to local organic_data.csv")
                self.df_full = self._ensure_dataframe(pd.read_csv("organic_data.csv"))
            else:
                raise e

    def run(self):
        # Identify Top 5 Influencers dynamically
        top_influencers = self.df_full['Name_x'].value_counts().head(5).index.tolist()
        print(f"Top 5 Influencers: {top_influencers}")
        
        # FOR VERIFICATION: Only process the first influencer
        top_influencers = top_influencers[:1]
        
        results = []
        detailed_rows = []
        
        for influencer in top_influencers:
            print(f"\n\n{'='*50}")
            print(f"Processing Influencer: {influencer}")
            print(f"{'='*50}")
            
            # 1. Prepare Data
            train_texts, pool_texts, sponsored_texts = self.prepare_data(influencer)
            
            if not train_texts:
                print(f"Skipping {influencer} (insufficient data)")
                continue
                
            # 2. Fine-Tune Generative Model (GPT-2 LoRA)
            gen_model_path = self.train_generator(influencer, train_texts)
            
            # 3. Fine-Tune Embedding Model (SimCSE/Contrastive)
            emb_model = self.train_embedding_model(influencer, train_texts)
            
            # 4. Generate Modified Posts
            print(f"Generating modified versions for {len(sponsored_texts)} sponsored posts...")
            modified_texts = self.generate_modified(gen_model_path, sponsored_texts, influencer, pool_texts)
            
            # 5. Compute KNN Distances
            metrics, per_post = self.evaluate_knn(emb_model, pool_texts, sponsored_texts, modified_texts)
            
            results.append({
                "Influencer": influencer,
                "Sponsored Count": len(sponsored_texts),
                "Avg Original Sim": metrics['avg_orig_sim'],
                "Avg Modified Sim": metrics['avg_mod_sim'],
                "Improvement": metrics['improvement']
            })

            for row in per_post:
                detailed_rows.append({
                    "Influencer": influencer,
                    "Original Sponsored": row["original"],
                    "Modified": row["modified"],
                    "Avg Original Sim": row["avg_orig_sim"],
                    "Avg Modified Sim": row["avg_mod_sim"],
                    "Improvement": row["improvement"],
                })
            
        # 6. Report
        results_df = pd.DataFrame(results)
        print("\n\n=== FINAL RESULTS TABLE ===")
        print(results_df)
        results_df.to_csv("final_comprehensive_report.csv", index=False)

        if detailed_rows:
            detailed_df = pd.DataFrame(detailed_rows)
            detailed_df.to_csv("final_results.csv", index=False)

    def prepare_data(self, influencer) -> tuple[list[str], list[str], list[str]]:
        # Filter by influencer
        df = self.df_full[self.df_full['Name_x'] == influencer].copy()
        
        # Sort (assuming index is somewhat chronological or 'Posts' count helps, but index is safest fallback)
        df = df.reset_index(drop=True)
        
        if len(df) < 150:
            print("Warning: Dataset too small.")
            return [], [], []
            
        # Split: Latest 100 vs Rest
        latest_100 = df.iloc[-100:].copy()
        train_df = df.iloc[:-100].copy()
        
        train_texts = [str(t) for t in train_df['caption'].dropna().tolist()]
        
        # Extract Sponsored from Latest 100
        # Use 'Sponsored' column if 1, else keywords
        if 'Sponsored' in latest_100.columns and latest_100['Sponsored'].sum() > 0:
            sponsored_df = latest_100[latest_100['Sponsored'] == 1]
        else:
            # Keyword fallback
            keywords = ['#ad', '#sponsored', 'sponsored', ' ad ', 'promo', 'partner', 'gifted', 'collab']
            pattern = '|'.join(keywords)
            latest_100['is_sponsored_guess'] = latest_100['caption'].apply(
                lambda x: bool(re.search(pattern, str(x).lower(), re.IGNORECASE))
            )
            sponsored_df = latest_100[latest_100['is_sponsored_guess'] == True]

        # Enforce 10-15 sponsored posts from latest 100
        # FOR VERIFICATION: Only process 3 sponsored posts
        SPONSORED_MAX_VERIF = 3
        if len(sponsored_df) > SPONSORED_MAX_VERIF:
            sponsored_df = sponsored_df.sample(n=SPONSORED_MAX_VERIF, random_state=42)
        elif len(sponsored_df) < SPONSORED_MIN:
            print("  Note: Few sponsored posts found. Sampling additional posts as 'Mock Sponsored' for pipeline consistency.")
            needed = SPONSORED_MIN - len(sponsored_df)
            remaining = latest_100[~latest_100.index.isin(sponsored_df.index)]
            if len(remaining) >= needed:
                extra = remaining.sample(n=needed, random_state=42)
                sponsored_df = pd.concat([sponsored_df, extra], axis=0)
            else:
                sponsored_df = latest_100.copy()
            # Still cap at 3 for verif
            sponsored_df = sponsored_df.head(SPONSORED_MAX_VERIF)
            
        sponsored_texts = [str(t) for t in sponsored_df['caption'].dropna().tolist()]
        
        # Ensure 'test_texts' (the pool) doesn't contain the sponsored ones (or does it? User says "Find 5 nearest non-sponsored")
        # So pool should be Purely Non-Sponsored
        pool_df = latest_100[~latest_100.index.isin(sponsored_df.index)]
        pool_texts = [str(t) for t in pool_df['caption'].dropna().tolist()]
        
        return train_texts, pool_texts, sponsored_texts

    def train_generator(self, influencer, train_texts):
        output_dir = os.path.join(OUTPUT_ROOT, f"{influencer}_gen")
        
        # Skip if exists
        if os.path.exists(os.path.join(output_dir, "adapter_model.safetensors")):
            print("  Generator already trained. Skipping.")
            return output_dir

        print("  Training Generator (GPT-2 LoRA)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_GEN)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_GEN)
        
        # LoRA Config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        
        # Dataset
        dataset = Dataset.from_dict({"text": train_texts})
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        # Adjust epochs based on device to prevent timeout
        epochs = 3 if self.device == "cuda" else 1
        print(f"  Training for {epochs} epochs (Device: {self.device})...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            num_train_epochs=epochs,
            learning_rate=5e-4,
            logging_steps=50,
            save_steps=500,
            report_to="none"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        
        trainer.train()
        model.save_pretrained(output_dir)
        return output_dir

    def train_embedding_model(self, influencer, train_texts):
        output_dir = os.path.join(OUTPUT_ROOT, f"{influencer}_emb")
        
        # Check if exists
        if os.path.exists(output_dir):
            print("  Embedding Model already trained. Loading...")
            try:
                return SentenceTransformer(output_dir)
            except:
                pass
            
        print("  Training Embedding Model (SimCSE)...")
        try:
            model = SentenceTransformer(MODEL_NAME_EMB)
        except Exception as e:
            print(f"  Error loading {MODEL_NAME_EMB}: {e}. Falling back to MiniLM.")
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Unsupervised SimCSE: InputExample(texts=[t, t]) with MultipleNegativesRankingLoss
        train_examples = [InputExample(texts=[t, t], label=1) for t in train_texts if len(t) > 20]
        
        if not train_examples:
            return model
            
        train_dataloader = DataLoader(_ListDataset(train_examples), shuffle=True, batch_size=16)
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
            output_path=output_dir,
            show_progress_bar=False
        )
        return model

    def generate_modified(self, model_path, sponsored_texts, influencer_name, pool_texts):
        modified_texts = []
        
        # Get style examples (top 10 random organic posts)
        import random
        style_examples = random.sample(pool_texts, min(len(pool_texts), 10))
        
        # Load embedding model for candidate selection
        try:
            emb_model = SentenceTransformer(MODEL_NAME_EMB)
        except Exception as e:
            print(f"  Error loading {MODEL_NAME_EMB}: {e}. Falling back to MiniLM.")
            emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
        pool_embeddings = emb_model.encode(pool_texts)

        # Prefer Ollama if available for stronger rewrites
        if self._ollama_available():
            for text in sponsored_texts:
                print(f"  Rewriting: {text[:50]}...")
                # Generate 3 candidates one by one to avoid timeouts
                candidates = []
                for i in range(3):
                    cand = self._generate_single_candidate_with_ollama(text, influencer_name, style_examples, seed=i)
                    if cand:
                        candidates.append(cand)
                
                if candidates:
                    # Select the one with lowest distance to organic pool
                    best_candidate = self._select_best_candidate(candidates, pool_embeddings, emb_model)
                    modified_texts.append(best_candidate)
                else:
                    modified_texts.append(self._light_clean(text))
            return modified_texts

        # Fallback to LoRA GPT-2
        for text in sponsored_texts:
            modified_texts.append(self._generate_with_gpt2(model_path, text))

        return modified_texts

    def _generate_single_candidate_with_ollama(self, text, influencer_name, style_examples, seed=0):
        """Generate ONE rewritten version using Ollama."""
        organic_posts_str = "\n---\n".join(style_examples)
        
        prompt = f"""
You are an expert social media manager work on: "Influencer Engagement Prediction and Creator-Consistent Rewriting"

Goal: Rewrite the given sponsored caption to match the influencer's style and reduce semantic distance to organic posts.

-----------------------------------------
STYLE EXAMPLES (Organic Posts from {influencer_name})
-----------------------------------------
{organic_posts_str}

-----------------------------------------
SPONSORED CAPTION
-----------------------------------------
{text}

-----------------------------------------
STRICT RULES
-----------------------------------------
1. Match influencer tone, emoji usage, and sentence length.
2. Keep product name and claims UNCHANGED.
3. Avoid marketing/corporate language (no "Introducing", "Experience", "Buy now").
4. Make it sound like a natural personal experience.

TASK:
Generate ONE rewritten caption. 
Return ONLY the rewritten text. No commentary.
(Variation {seed})
"""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7 + (seed * 0.1)
        }

        try:
            # Increased timeout to 180s for Mistral
            response = requests.post(OLLAMA_URL, json=payload, timeout=180)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"    Error generating candidate: {e}")
        
        return None

    def _select_best_candidate(self, candidates, pool_embeddings, emb_model):
        """Select the candidate with the highest similarity (lowest distance) to the organic pool."""
        if not candidates:
            return ""
            
        cand_embeddings = emb_model.encode(candidates)
        # Similarity to all organic posts
        sim_matrix = cosine_similarity(cand_embeddings, pool_embeddings)
        # Average of top-5 similarity for each candidate
        topk_sims = np.sort(sim_matrix, axis=1)[:, -KNN_K:]
        avg_sims = np.mean(topk_sims, axis=1)
        
        # Pick index with maximum average similarity
        best_idx = np.argmax(avg_sims)
        return candidates[best_idx]

    def _get_gpt2_components(self, model_path):
        if getattr(self, "_gpt2_cache", None) is None:
            self._gpt2_cache = {}
        if model_path not in self._gpt2_cache:
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_GEN)
            model = PeftModel.from_pretrained(base_model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_GEN)
            tokenizer.pad_token = tokenizer.eos_token
            model.to(self.device)
            model.eval()
            self._gpt2_cache[model_path] = (model, tokenizer)
        return self._gpt2_cache[model_path]

    def _generate_with_gpt2(self, model_path, text):
        model, tokenizer = self._get_gpt2_components(model_path)

        # Seed strategy: First sentence + " "
        sentences = re.split(r'[.!?\n]', text)
        seed = sentences[0]
        if len(seed) < 20 and len(sentences) > 1:
            seed += ". " + sentences[1]

        inputs = tokenizer(seed, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )

        candidate = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return candidate

    def _light_clean(self, text):
        cleaned = re.sub(r"https?://\S+", "", text)
        cleaned = re.sub(r"\s+#(ad|sponsored|partner|promo|gifted|collab)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _ollama_available(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def evaluate_knn(self, model, pool_texts, original_sponsored, modified_texts):
        # Embed Pool
        pool_embeddings = model.encode(pool_texts)

        # Embed Queries
        orig_embeddings = model.encode(original_sponsored)
        mod_embeddings = model.encode(modified_texts)

        # Cosine similarity to all organic pool posts
        orig_sims = cosine_similarity(orig_embeddings, pool_embeddings)
        mod_sims = cosine_similarity(mod_embeddings, pool_embeddings)

        # Average of top-K similarities
        orig_topk = np.sort(orig_sims, axis=1)[:, -KNN_K:]
        mod_topk = np.sort(mod_sims, axis=1)[:, -KNN_K:]

        avg_orig = float(np.mean(orig_topk))
        avg_mod = float(np.mean(mod_topk))

        per_post = []
        orig_avg_per = np.mean(orig_topk, axis=1)
        mod_avg_per = np.mean(mod_topk, axis=1)
        for original, modified, o_sim, m_sim in zip(original_sponsored, modified_texts, orig_avg_per, mod_avg_per):
            per_post.append({
                "original": original,
                "modified": modified,
                "avg_orig_sim": float(o_sim),
                "avg_mod_sim": float(m_sim),
                "improvement": float(m_sim - o_sim),
            })

        return {
            "avg_orig_sim": avg_orig,
            "avg_mod_sim": avg_mod,
            "improvement": avg_mod - avg_orig
        }, per_post

    def _ensure_dataframe(self, df_like):
        if isinstance(df_like, pd.DataFrame):
            return df_like
        if isinstance(df_like, Iterator):
            frames = list(df_like)
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(df_like)


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

if __name__ == "__main__":
    pipeline = ComprehensivePipeline()
    pipeline.run()
