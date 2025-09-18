# ReviewRecommender

ReviewRecommender is a machine learning-powered semantic search and recommendation system for Yelp reviews. Enter a query like “good pasta” or “great haircut” and instantly get the most relevant, emotionally aware reviews about local shops and restaurants—leveraging vector database search and state-of-the-art NLP.

## Features

- Semantic review search: Finds reviews closely matching the user’s intent using sentence embeddings and vector search (ChromaDB).
- Emotion filtering: Classifies and highlights reviews by mood (happy, angry, surprising, sad, etc.) with Hugging Face Transformers.
- Category filtering: Search within restaurants, services, or all businesses.
- Interactive Gradio dashboard: Clean UI for trying out queries and filters visually.
- Built on a large, real-world dataset (Yelp public reviews).

## Tech Stack

`["Python", "pandas", "LangChain", "ChromaDB", "Gradio", "Hugging Face Transformers", "Sentence Transformers", "OpenAI API", "Zero-Shot Classification", "Emotion Analysis", "Vector Database", "NLP", "Semantic Search", "python-dotenv", "Matplotlib", "Seaborn"]`

## Dataset

- Source: [Yelp Open Dataset](https://www.yelp.com/dataset)
- Data loaded, filtered, and enriched—including reviews, business info, categories, locations, and emotional tags.
- Local data is not shipped with the repo; see usage instructions to download and preprocess.

## How to Run

1. **Clone this repository**  
   `git clone ... && cd ReviewRecommender`

2. **Install dependencies**  
   `pip install -r requirements.txt`

3. **Get and prepare the dataset:**  
   - Download the Yelp dataset ([Yelp Open Dataset](https://www.yelp.com/dataset)).
   - Use provided scripts or code snippets (`part1.ipynb`/`part2.ipynb`) to preprocess reviews and business info.
   - Ensure `cleaned_reviews_with_emotions.csv` is present in your repo directory.

4. **Set up environment variables**  
   - If using APIs, place your keys in a `.env` file.

5. **Run the Gradio dashboard**  
   - `python app.py` or run the provided sample code in Jupyter/notebook environment.

## Example Usage

- User enters: `authentic Thai food`  
- Selects category: `Restaurants`  
- Selects tone: `happy`  
- System returns top 5-10 reviews for local Thai restaurants with highest "happy" emotion scores and semantic similarity.

## Observations

- Embedding-based search significantly outperforms keyword search for nuanced queries.
- Emotion analysis adds an extra dimension, letting users find reviews that match not just what but how they want to feel.
- Category and tone filtering increases user trust and recommendation relevance.

## File Structure

- `part1.ipynb` — Data download, cleaning, merging, and feature engineering
- `part2.ipynb` — NLP, embeddings, vector search, emotion model, Gradio UI
- `requirements.txt` — Library dependencies
- `.gitignore` — Ignore temporary/data files
- `images/` — (Optional) Example dashboard/screenshots

## License

MIT License

## Credits

Inspired by open-source tutorials and the Yelp Open Dataset. Built using pandas, LangChain, Chroma, Hugging Face Transformers, Gradio, and more.

---

**Ready to make your Yelp review search smart, semantic, and mood-aware!**
