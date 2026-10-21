# 🎬 CineMind-AI

**CineMind-AI** is a content-based movie recommendation web app built with Python and Streamlit. Select a movie you like and discover similar movies in a Netflix-inspired poster grid featuring similarity scores, ratings, release years, and genres.

**🚀 Live Demo:** [Try CineMind-AI](https://purvi-cinemind-ai.streamlit.app/)


## ✨ Features

- **Content-Based Recommendations:** Discover similar movies from a dataset containing 45,000+ movie records using TF-IDF vectorization and cosine similarity.
- **Netflix-Inspired Interface:** Enjoy a dark theme, movie poster cards, and plot summaries on hover.
- **Similarity Scores and Ratings:** View the similarity score, movie rating, release year, and genres for each recommendation.
- **Genre Highlighting:** Identify genres shared with your selected movie.
- **Customizable Filters:** Adjust the number of recommendations (4–24), minimum rating, genres, and sorting criteria.
- **Fast Movie Search:** Search popular titles through the main dropdown or use the full-dataset search feature.
- **Surprise Me:** Discover a random movie with a single click.
- **Updated Movie Posters:** Fetch poster images from the TMDB API instead of relying on potentially outdated poster links in the dataset.

## 🧠 How It Works

CineMind-AI uses content-based filtering to recommend movies based on their textual metadata.

1. **Text Preparation:** Each movie's text representation combines available information, such as its overview, tagline, and genres.
2. **TF-IDF Vectorization:** The text is converted into numerical vectors using TF-IDF with up to 50,000 features.
3. **Similarity Calculation:** When a movie is selected, cosine similarity compares its TF-IDF vector with those of other movies.
4. **Ranking and Filtering:** Duplicate entries are removed, the selected filters are applied, and the remaining movies are ranked according to the selected sorting option.
5. **Poster Retrieval:** Movie posters are fetched from TMDB using the available movie identifiers.

### 📊 Understanding the Match Score

The displayed match percentage represents cosine similarity multiplied by 100. For example, a cosine similarity score of 0.25 is displayed as 25%.

These scores measure textual similarity, not the probability that you will enjoy a movie. They depend on the dataset and text-processing pipeline and are best used to compare recommendations within the app.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Data Processing:** pandas, NumPy
- **Machine Learning:** scikit-learn
- **Sparse Matrix Operations:** SciPy
- **Model and Data Persistence:** joblib
- **Movie Posters:** TMDB API

## 📁 Project Structure

```text
CineMind-AI/
├── .streamlit/
│   └── config.toml
├── model_artifacts/
│   ├── movies_data.joblib
│   └── tfidf_matrix.joblib
├── app.py
├── movies_metadata.csv
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** Create `.streamlit/secrets.toml` locally to store your TMDB API key. Do not upload this file to GitHub. Configure the key separately in Streamlit Community Cloud when deploying.

## 🚀 Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/PurviGarg/CineMind-AI.git
cd CineMind-AI
```

### 2. Create and Activate a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the TMDB API Key

1. Create a free account on [The Movie Database (TMDB)](https://www.themoviedb.org/).
2. Open your account settings, navigate to the API section, and request an API key.
3. Create a file named `.streamlit/secrets.toml` in the project directory.
4. Add the following configuration:

```toml
TMDB_API_KEY = "your_api_key_here"
```

Keep this file private and ensure `.streamlit/secrets.toml` is listed in `.gitignore`. Never commit your actual API key to GitHub.

If the application supports running without an API key, it can still be used without poster images.

### 5. Start the Application

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## ☁️ Deploy on Streamlit Community Cloud

1. Push your project to [GitHub](https://github.com/), ensuring that `.streamlit/secrets.toml` and other sensitive files are excluded.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select your repository, branch (`main`, if applicable), and entry-point file (`app.py`).
4. Open the app's advanced settings and add your API key under **Secrets**:

```toml
TMDB_API_KEY = "your_api_key_here"
```

5. Deploy the application and wait for the build to finish.
6. Open the deployed app and test the movie search, recommendations, filters, and poster loading.

## ⚠️ Limitations

- **Content-Based Recommendations:** Recommendations are based on movie metadata, not individual viewing history or user preferences.
- **Similarity Scores:** A high score indicates textual similarity, not guaranteed enjoyment or recommendation accuracy.
- **Search Performance:** The main dropdown contains a curated selection of approximately 4,000 popular movies to keep the interface responsive. The full-dataset search feature is intended to cover the remaining records.
- **Poster Availability:** Some movies may not have an available poster on TMDB.
- **API Dependency:** Poster retrieval depends on TMDB API availability, network connectivity, and valid API credentials.

## 🙌 Credits

- **Movie Dataset:** [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- **Movie Posters and Metadata:** [The Movie Database (TMDB)](https://www.themoviedb.org/)

This product uses the TMDB API but is not endorsed or certified by TMDB.

## 👩‍💻 Author

**Purvi Garg**

Built as an educational machine learning project to explore content-based filtering, text vectorization, cosine similarity, and interactive web application development.

## 📄 License

This project is presented as an educational project. If you want others to reuse, modify, or distribute the code under defined terms, add a `LICENSE` file with your chosen license, such as the MIT License.
