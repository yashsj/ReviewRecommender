#%%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")

print("Path to dataset files:", path)
#%%
import os
print(os.listdir())
#%%
#/Users/suyashj/.cache/kagglehub/datasets/yelp-dataset/yelp-dataset/versions/4
import json
import random
import pandas as pd

# First, count lines (total reviews)
with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

# Randomly select line numbers to sample
sample_size = 12000
random.seed(42)
selected_lines = set(random.sample(range(total_lines), sample_size))

# Read only those lines
reviews = []
with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i in selected_lines:
            reviews.append(json.loads(line))
        if len(reviews) == sample_size:
            break

df_reviews = pd.DataFrame(reviews)
print(df_reviews.head())

#%%
import json

# Get unique business_ids from your reviews
business_ids = set(df_reviews['business_id'])

# Read business.json and keep only the matching businesses
businesses = []
with open('yelp_academic_dataset_business.json', 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        if record['business_id'] in business_ids:
            businesses.append(record)

df_business = pd.DataFrame(businesses)
print(df_business.head())

#%%
df_merged = pd.merge(df_reviews, df_business, on='business_id', how='left')
print(df_merged.head())
#%%
df_merged
#%%
import seaborn as sns
import matplotlib.pyplot as plt
#%%
ax=plt.axes()
sns.heatmap(df_merged.isna().transpose(),cbar=False,ax=ax)
plt.xlabel("Columns")
plt.ylabel("Missing values")
#%%
import numpy as np

# Assuming your DataFrame is named df_merged and has a 'date' column
df_merged['review_date'] = pd.to_datetime(df_merged['date'])
df_merged['review_year'] = df_merged['review_date'].dt.year

# Preview the new column
print(df_merged[['date', 'review_year']].head())
#%%
#Before even having the corr matrix lets rename those cols
df_merged.rename(
    columns={
        'stars_x': 'review_stars',   # Rename individual review ratings
        'stars_y': 'business_avg_stars'  # Rename business average ratings
    },
    inplace=True  # Modify the DataFrame directly
)

# Verify the changes
print(df_merged.columns.tolist())
#%%
#We only want to consider the following cols
columns_of_interest=["business_avg_stars", "review_stars","review_count","review_year"]
#Build the correlation matrix
correlation_matrix=df_merged[columns_of_interest].corr(method="spearman")
sns.set_theme(style="white")
plt.figure(figsize=(10,10))
heapmap=sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm",cbar_kws={"label":"Spearman Correlation"})
plt.show()
#%%
df_merged[(df_merged['review_count'].isna() | df_merged['review_year'].isna()) | (df_merged['review_stars'].isna())]
#%%
df_merged
#%%
df_merged["categories"].value_counts().reset_index().sort_values(by="count", ascending=False)
#%%
df_merged["words_in_description"]=df_merged["text"].str.split().str.len()
#%%
df_merged["words_in_description"].mean
#%%
df_merged
#%%
df_merged.loc[df_merged["words_in_description"].between(15,24),"text"]
#%%
df_merged_25_words=df_merged[df_merged["words_in_description"]>=25]
#%%
df_merged_25_words
#%%
#Merge the text and review_id
df_merged_25_words['text_with_id'] = df_merged_25_words['text'] + " [ID: " + df_merged_25_words['review_id'] + "]"

#%%
df_merged_25_words
#%%
# List of columns to keep
columns_to_keep = [
    'review_id',
    'text',  # or 'text_with_id' if you merged them
    'review_stars',
    'business_avg_stars',
    'review_year',
    'categories',
    'is_open',
    'city',
    'state',
    'name',
    'business_id',
    'words_in_description'
]

# Create a new DataFrame with only these columns
df_cleaned = df_merged[columns_to_keep].copy()
#%%
df_cleaned.to_csv('cleaned_reviews.csv', index=False)

#%%
print(df_cleaned.info())
#%%
print(df_cleaned.describe())
#%%
