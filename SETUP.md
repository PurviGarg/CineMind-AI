# MovieMatch Streamlit frontend

## 1. Save model files from your notebook

Run this cell **after** you have finished preprocessing `df` and fitting `tfidf_matrix`.
Your `df` must have the same row order as `tfidf_matrix` and must include the processed `genres`, `tags`, and `title` columns.

```python
import joblib
from pathlib import Path

artifact_dir = Path("model_artifacts")
artifact_dir.mkdir(exist_ok=True)

joblib.dump(df.reset_index(drop=True), artifact_dir / "movies_data.joblib")
joblib.dump(tfidf_matrix, artifact_dir / "tfidf_matrix.joblib")

print("Saved files:", list(artifact_dir.iterdir()))
```

Important: save the exact `df` used to create `tfidf_matrix`. Do not sort, filter, or reset the dataframe differently after fitting the matrix.

## 2. Put files in one project folder

Your folder should look like this:

```text
MovieRecommendationSystem/
├── app.py
├── requirements.txt
└── model_artifacts/
    ├── movies_data.joblib
    └── tfidf_matrix.joblib
```

If you use Google Colab, download the two saved files/folder and place them in this project folder on your computer.

## 3. Install and launch

Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit should open the app in your browser. If it does not, open the local URL printed in the terminal.

## Included features

- Search a movie by title (case-insensitive; partial matches show a selection list).
- Filter by minimum rating and genre.
- Choose how many recommendations to display.
- Results shown in cards and a table.
- Movie posters are shown if your saved dataframe contains valid `poster_path` values.

The app uses the same general ranking approach discussed in the notebook: 65% text similarity, 25% genre overlap, and 10% rating score. These weights are starter values, not experimentally optimized.
