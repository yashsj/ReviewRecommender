from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import collections
from sentence_transformers import SentenceTransformer
from tomlkit import string

#%%
from dotenv import load_dotenv
load_dotenv()
#%%
import pandas as pd
#%%
data=pd.read_csv('cleaned_reviews.csv')


#%%
data
#%%
data['review_id_text'] = data['review_id'] + " " + data['text']
#%%
data['review_id_text'].head()
#%%

#%%

#%%
pwd
#%%
#Load into the TextLoader,
raw_documents=TextLoader("cleaned_reviews.txt").load()
text_splitter1=CharacterTextSplitter(chunk_size=0,chunk_overlap=0,separator="\n")
documents=text_splitter1.split_documents(raw_documents)
#%%
documents[0]
#%%

from sentence_transformers import SentenceTransformer


class SBERTEmbeddingFunction:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    def __call__(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()


#%%
embedder = SentenceTransformer('all-MiniLM-L6-v2')
#%%
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db_reviews = Chroma.from_documents(
    documents,
    embedding=embeddings,
)

#%%

#%%
# 4. Query using Chroma's built-in methods (no need for manual embedding)
query = "Haircut"
results = db_reviews.similarity_search(query, k=5)  # Get top 5 matches

# 5. Print results
print("Top 5 matching reviews:")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
#%%
data[data["review_id"]==str(documents[0].page_content.split()[0].strip())]
#%%
def retreive_semantic_recommendations(query:str, k:int)->pd.DataFrame:
    recs=db_reviews.similarity_search(query, k=10)
    review_list=[]
    for i in range(len(recs)):
        review_list+=[str(recs[i].page_content.split()[0].strip())]

    return data[data["review_id"].isin(review_list)].head(k)

#%%
retreive_semantic_recommendations("Indian food",10)
#%%
