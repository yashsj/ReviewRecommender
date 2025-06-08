from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

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
ls -a
#%%
pwd
#%%
#Load into the TextLoader,
raw_documents=TextLoader("cleaned_reviews.txt").load()
text_splitter=CharacterTextSplitter(chunk_size=0,chunk_overlap=0,seperator="\n")
