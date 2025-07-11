# Load the libs
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
import gradio as gr

load_dotenv()
data = pd.read_csv("cleaned_reviews_with_emotions.csv")

# Build Chroma documents with review_id in metadata
documents = [
    Document(
        page_content=row["text"],
        metadata={"review_id": row["review_id"]}
    )
    for _, row in data.iterrows()
]

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db_reviews = Chroma.from_documents(
    documents,
    embedding=embeddings,
)

# Tone mapping: UI value -> DataFrame column
tone_column_map = {
    "happy": "happiness",
    "surprising": "surprise",
    "angry": "anger",
    "suspensful": "suspense",
    "sadness": "sadness"
}

def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None, initial_top_k: int = 50, final_top_k: int = 10) -> pd.DataFrame:
    recs = db_reviews.similarity_search(query, k=initial_top_k)
    rec_ids = [rec.metadata["review_id"] for rec in recs]
    review_recs = data[data["review_id"].isin(rec_ids)].copy()

    # Optional: filter by category if not "All"
    if category and category != "All":
        review_recs = review_recs[review_recs["simple_category"] == category]

    # Sort by tone if specified
    if tone and tone in tone_column_map:
        review_recs = review_recs.sort_values(by=tone_column_map[tone], ascending=False)

    return review_recs.head(final_top_k)

def recommend_reviews(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        review_id = row["review_id"]
        business = row.get("business_name", "Unknown Business")  # If you have business_name column
        text = row["text"]
        truncated = " ".join(text.split()[:30]) + "..." if len(text.split()) > 30 else text
        rating = row.get("stars", "N/A")
        location = f"{row.get('city', '')}, {row.get('state', '')}"
        categories = row.get("categories", "")
        # Markdown card for each review
        card = f"""**{business}**  \n⭐ {rating}  \n*{categories}*  \n_{location}_  \n\n> {truncated}\n\nID: `{review_id}`"""
        results.append(card)
    return results

categories = ["All"] + sorted(data['simple_category'].dropna().unique())
tones = ["All", "happy", "surprising", "angry", "suspensful", "sadness"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    with gr.Row():
        user_query = gr.Textbox(label="Please enter a business/restaurant", placeholder="a great Pizza Place")
        category_dropdown = gr.Dropdown(choices=categories, label="Categories", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select Tone", value="All")
        submit_button = gr.Button(value="Find Recommendations")

    gr.Markdown("## Recommendations")
    output= gr.Markdown(label="Recommendations")
  # Or gr.Dataframe, or gr.Markdown in a loop

    def display_results(*args):
        results = recommend_reviews(*args)
        # Join all cards into one Markdown block for display
        return "\n\n---\n\n".join(results)

    submit_button.click(fn=display_results,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()
