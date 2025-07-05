import pandas as pd

#%%
data=pd.read_csv('cleaned_reviews.csv')
#%%
data.head()
#%%
data['categories'].value_counts().reset_index()
#%%
filtered_data = data[data['categories'].str.contains('Restaurants',na=False)]

#%%
filtered_data
#%%
non_restaurants = data[~data['categories'].str.contains('Restaurants', na=False)]
#%%
non_restaurants
#%%
restaurants = data[data['categories'].str.contains('Restaurants', na=False)]
unique_restaurant_categories = restaurants['categories'].unique().tolist()
restaurant_mapping = {cat: 'Restaurants' for cat in unique_restaurant_categories}

#%%
non_restaurants = data[~data['categories'].str.contains('Restaurants', na=False)]
services_keywords = ['Service', 'Services', 'Repair', 'Cleaning', 'Laundry', 'Plumbing', 'Car Wash', 'Auto']

def assign_services_group(cat):
    if not isinstance(cat, str):
        return 'Services'
    for keyword in services_keywords:
        if keyword.lower() in cat.lower():
            return 'Services'
    return 'Services'

non_restaurants_mapping = {cat: assign_services_group(cat) for cat in non_restaurants['categories'].unique()}

#%%
final_category_mapping = {**restaurant_mapping, **non_restaurants_mapping}

#%%
data['simple_category'] = data['categories'].map(final_category_mapping)
#%%
data
#%%
from transformers import pipeline
categories=["Restaurants","Services"]
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device="mps")

#%%
sequence=data.loc[data['simple_category']=='Restaurants', 'text'].reset_index(drop=True)[0]
#%%
sequence
#%%
pipe(sequence,categories)
#%%
import numpy as np
max_index=np.argmax(pipe(sequence,categories)["scores"])
max_label=pipe(sequence,categories)["labels"][max_index]
#%%
max_label
#%%
def generate_preditions(sequence,categories):
    predictions=pipe(sequence,categories)
    max_index=np.argmax(pipe(sequence,categories)["scores"])
    max_label=pipe(sequence,categories)["labels"][max_index]
    return max_label
#%%
from tqdm import tqdm
actual_cats=[]
predicted_cats=[]

for i in tqdm(range(0, 300)):
    sequence=data.loc[data['simple_category']=='Restaurants', 'text'].reset_index(drop=True)[i]
    predicted_cats+=[generate_preditions(sequence,categories)]
    actual_cats+=['Restaurants']
#%%
for i in tqdm(range(0, 300)):
    sequence=data.loc[data['simple_category']=='Services', 'text'].reset_index(drop=True)[i]
    predicted_cats+=[generate_preditions(sequence,categories)]
    actual_cats+=['Services']
#%%
predictions_df=pd.DataFrame({"actual_cateogries":actual_cats,"predicted_categories":predicted_cats})
#%%
predictions_df
#%%
predictions_df["correction_predicton"]=np.where(predictions_df["actual_cateogries"]==predictions_df["predicted_categories"],1,0)
#%%
predictions_df
#%%
predictions_df["correction_predicton"].sum()/len(predictions_df)
#%%
data#Create two
#%%
from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",top_k=None,device="mps")
classifier("I love this!")

#%%
data["text"][4]
#%%
classifier(data["text"][4].split("."))
#%%
sentences=data["text"][0].split(".")
predictions=classifier(sentences)
#%%
predictions[0]
#%%

#%%
#Calculate the emotional scores for each sentence for the review and get the max whcih will be the overall emotional label
import numpy as np
emotion_labels=["anger","disgust","fear","happiness","sadness","surprise"]
review_id=[]
emotion_scores={label:[] for label in emotion_labels}

def calculate_emotion_scores(predictions):
    per_emotion_scores={label:[] for label in emotion_labels}
    for prediction in predictions:
        sorted_predictions=sorted(prediction,key=lambda x:x["label"])
        for index, label in enumerate(emotion_labels):
            per_emotion_scores[label].append(sorted_predictions[index]["score"])
    return {label:np.max(scores) for label,scores in per_emotion_scores.items()}


#%%
for i in range(0,10):
    review_id=data["text"][i]
    sentences=data["text"][i].split(".")
    predictions=classifier(sentences)
    max_scores=calculate_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])

#%%
emotion_scores
#%%
from tqdm import tqdm


emotion_labels=["anger","disgust","fear","happiness","sadness","surprise"]
review_id=[]
emotion_scores={label:[] for label in emotion_labels}


for i in tqdm(range(len(data))):
    review_id=data["text"][i]
    sentences=data["text"][i].split(".")
    predictions=classifier(sentences)
    max_scores=calculate_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])

#%%
emotions_df=pd.DataFrame(emotion_scores)
emotions_df["review_id"]=review_id
