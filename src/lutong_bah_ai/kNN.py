from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"

csv_path = (
    DATASET_DIR / "recipeex001.csv"
    if (DATASET_DIR / "recipeex001.csv").exists()
    else DATASET_DIR / "recipeex001"
)

df = pd.read_csv(csv_path)

print("Available columns:", list(df.columns))

# Adjust these column names if needed based on what printed above
TITLE_COL = "Dish Name" if "Dish Name" in df.columns else df.columns[0]
ING_COL = (
    "Ingredients"
    if "Ingredients" in df.columns
    else next(
        (c for c in df.columns if "ingredient" in c.lower()), df.columns[1]
    )
)

# Clean missing values and normalize text
df = df.copy()
df[TITLE_COL] = df[TITLE_COL].fillna("Unnamed Recipe").astype(str)
df[ING_COL] = (
    df[ING_COL]
    .fillna("")
    .astype(str)
    .str.replace(",", " ", regex=False)
    .str.lower()
)

# Step 1: Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_points = vectorizer.fit_transform(df[ING_COL])

print(
    f"Created {tfidf_points.shape[0]} recipe points across"
    f" {tfidf_points.shape[1]} ingredient features."
)

# Step 2: Fit k-NN models for Euclidean, Manhattan, and Cosine
k = 5
models = {
    "Euclidean": NearestNeighbors(n_neighbors=k, metric="euclidean").fit(
        tfidf_points
    ),
    "Manhattan": NearestNeighbors(n_neighbors=k, metric="manhattan").fit(
        tfidf_points
    ),
    "Cosine": NearestNeighbors(n_neighbors=k, metric="cosine").fit(
        tfidf_points
    ),
}


def highlight_ingredients(
    text: str, query_terms: list[str], max_len: int = 120
) -> str:
  """Shorten long ingredient text and bold words matching the user's query."""
  snippet = text[:max_len] + ("..." if len(text) > max_len else "")
  for term in sorted(set(query_terms), key=len, reverse=True):
    if term:
      pattern = re.compile(rf"\b({re.escape(term)})\b", flags=re.IGNORECASE)
      snippet = pattern.sub(r"**\1**", snippet)
  return snippet.replace("|", "/")


def print_metric_table(
    label: str, indices_row, distances_row, query_terms: list[str], is_cosine: bool = False
) -> None:
  divider = "-" * 31
  print(f"\n{label} Table:")
  for rank, (row_idx, dist) in enumerate(zip(indices_row, distances_row), start=1):
    recipe_name = str(df.iloc[row_idx][TITLE_COL]).replace("|", "/")
    recipe_ing = str(df.iloc[row_idx][ING_COL])
    bolded_ing = highlight_ingredients(recipe_ing, query_terms)

    if is_cosine:
      similarity = 1.0 - float(dist)
      score_col = f"sim: {similarity:.4f} | dist: {dist:.4f}"
    else:
      score_col = f"dist: {dist:.4f}"

    print(divider)
    print(f"[{rank}] {recipe_name} | {score_col} | {bolded_ing}")
    print(divider)
    print()


# Step 3: Output Generation for User Query
raw_query = input("\nEnter ingredients (e.g., chicken, garlic, salt): ").strip()

if not raw_query:
  print("No ingredients entered. Exiting.")
else:
  cleaned_query = raw_query.replace(",", " ").lower()
  query_terms = [w for w in cleaned_query.split() if w]
  query_point = vectorizer.transform([cleaned_query])

  print(f"\nTop {k} recipes for query '{raw_query}':")

  for metric_name, model in models.items():
    distances, indices = model.kneighbors(query_point)
    print_metric_table(
        label=metric_name,
        indices_row=indices[0],
        distances_row=distances[0],
        query_terms=query_terms,
        is_cosine=(metric_name == "Cosine"),
    )
