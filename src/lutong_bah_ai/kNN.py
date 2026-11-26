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
df["RawIngredients"] = df[ING_COL].fillna("").astype(str)
df[ING_COL] = (
    df["RawIngredients"]
    .str.replace(",", " ", regex=False)
    .str.lower()
)

# Step 1: Vectorization (TF-IDF)
vectorizer_norm2 = TfidfVectorizer(stop_words="english", norm="l2")
vectorizer_norm1 = TfidfVectorizer(stop_words="english", norm="l1")
tfidf_points_norm2 = vectorizer_norm2.fit_transform(df[ING_COL])
tfidf_points_norm1 = vectorizer_norm1.fit_transform(df[ING_COL])

print(
    f"Created {tfidf_points_norm2.shape[0]} recipe points across"
    f" {tfidf_points_norm2.shape[1]} ingredient features."
)

# Step 2: Fit k-NN models for Euclidean, Manhattan, and Cosine
k = 5
models = {
    "Euclidean": NearestNeighbors(n_neighbors=k, metric="euclidean").fit(
        tfidf_points_norm2
    ),
    "Manhattan": NearestNeighbors(n_neighbors=k, metric="manhattan").fit(
        tfidf_points_norm1
    ),
    "Cosine": NearestNeighbors(n_neighbors=k, metric="cosine").fit(
        tfidf_points_norm2
    ),
}


def highlight_ingredients(text: str, query_terms: list[str]) -> str:
  """Highlight all words matching the user's query across the full ingredient list."""
  # Add a space after commas for cleaner reading
  formatted = re.sub(r",\s*", ", ", text.strip())
  for term in sorted(set(query_terms), key=len, reverse=True):
    if term:
      pattern = re.compile(rf"\b({re.escape(term)})\b", flags=re.IGNORECASE)
      formatted = pattern.sub(r"**\1**", formatted)
  return formatted


def print_metric_table(
    label: str, indices_row, distances_row, query_terms: list[str], is_cosine: bool = False
) -> None:
  header_line = "=" * 70
  divider = "-" * 70

  print(f"\n{header_line}")
  print(f"  {label.upper()} DISTANCE RESULTS")
  print(header_line)

  for rank, (row_idx, dist) in enumerate(zip(indices_row, distances_row), start=1):
    recipe_name = str(df.iloc[row_idx][TITLE_COL]).strip()
    recipe_ing = str(df.iloc[row_idx]["RawIngredients"])
    bolded_ing = highlight_ingredients(recipe_ing, query_terms)

    if is_cosine:
      similarity = 1.0 - float(dist)
      score_col = f"Similarity: {similarity:.4f}  |  Distance: {dist:.4f}"
    else:
      score_col = f"Distance: {dist:.4f}"

    print(f"\n[{rank}] {recipe_name}")
    print(f"    {score_col}")
    print(f"    Ingredients: {bolded_ing}")
    print(divider)


# Step 3: Output Generation for User Query
raw_query = input("\nEnter ingredients (e.g., chicken, garlic, salt): ").strip()

if not raw_query:
  print("No ingredients entered. Exiting.")
else:
  cleaned_query = raw_query.replace(",", " ").lower()
  query_terms = [w for w in cleaned_query.split() if w]

  print(f"\nTop {k} recipes for query '{raw_query}':")

  for metric_name, model in models.items():
    if metric_name == "Manhattan":
      query_point = vectorizer_norm1.transform([cleaned_query])
    else:
      query_point = vectorizer_norm2.transform([cleaned_query])

    distances, indices = model.kneighbors(query_point)
    print_metric_table(
        label=metric_name,
        indices_row=indices[0],
        distances_row=distances[0],
        query_terms=query_terms,
        is_cosine=(metric_name == "Cosine"),
    )
