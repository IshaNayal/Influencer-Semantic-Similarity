import os
import pandas as pd
from datasets import load_dataset

keywords = ['#ad', '#sponsored', '#partner', '#gifted', '#collab', ' ad ', 'promo', 'partner', 'sponsored']
pattern = '|'.join(keywords)

def load_df():
    if os.path.exists('organic_data.csv'):
        print('Loading organic_data.csv')
        return pd.read_csv('organic_data.csv')
    print('Loading HF dataset')
    ds = load_dataset('ishanayal16/influencer_data')
    return ds['train'].to_pandas()

df = load_df()

if 'Name_x' in df.columns:
    df['influencer'] = df['Name_x']

influencers = df['influencer'].value_counts().head(5).index.tolist()
print('Top 5:', influencers)

for name in influencers:
    subset = df[df['influencer'] == name]
    if 'caption' not in subset.columns:
        print('No caption column for', name)
        continue
    sponsored = subset[subset['caption'].str.contains(pattern, case=False, na=False)]
    picks = sponsored.head(2)
    if len(picks) < 2:
        picks = subset.head(2)
    print('\n===', name, '===')
    for i, text in enumerate(picks['caption'].tolist(), 1):
        print(f'[{i}] {text}\n')
