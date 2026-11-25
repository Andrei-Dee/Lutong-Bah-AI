from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import time

# https://docs.python.org/3/library/pathlib.html
# __file__ gives the path to this current file
# .parents returns a list of parents and we choose 2 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# then we select the dataset folder
DATASET_DIR = PROJECT_ROOT / "dataset"

# get the csv path
csv_path = (
    DATASET_DIR / "recipeex001.csv"
)

# we use dataframe to make it easier
df = pd.read_csv(csv_path)

# we load the SBERT model
# https://sbert.net/docs/quickstart.html#sentence-transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# we clean the data first, make sure na string lang yung input
# replace NaN or None values with string "Unnamed Recipe"
df["Name"] = df["Name"].fillna("Unnamed Recipe").astype(str)
# replace NaN or None values with empty string
df["Ingredients"] = df["Ingredients"].fillna("").astype(str).str.lower()

# we then convert to list those ingredients per recipe
recipe_texts = df["Ingredients"].tolist()


# https://sbert.net/docs/package_reference/sentence_transformer/model.html#sentence_transformers.sentence_transformer.model.SentenceTransformer.encode
# then we encode into vector embeddings the recipe_texts
# btw it returns an ndarray (sa numpy)
# with shape [num_inputs, output_dimension] where in this case
# since all-MiniLM-L6-v2 gamit natin, 384 output_dimensions
# as mentioned here https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
recipe_embeddings = model.encode(
    recipe_texts,
    # note this is not the same in chapter 16 mini-batching
    # since this is already pretrained, no weight adjustments is happening
    # it simply means it process 64 recipes at a time
    batch_size=64,
    show_progress_bar=True,
    # set to true to normalize returned vectors to have length 1
    # makes cosine similarity easier
    # read more on the docs
    normalize_embeddings=True,
)

# ask the user for ingredients
raw_query = input("\nEnter ingredients (e.g., chicken, garlic, soy sauce): ").strip()

if raw_query:
    # https://www.w3schools.com/python/ref_module_time.asp
    # start the timer
    start_time = time.perf_counter()

    # get that one vector we embedded
    query_vector = model.encode([raw_query.lower()], normalize_embeddings=True)[0]

    # cosine similarity (since vectors are normalized, dot product = cosine similarity!)
    scores = np.dot(recipe_embeddings, query_vector)

    # sort scores from highest to lowest, and grab the top 5 row numbers
    top_indices = np.argsort(scores)[::-1][:5]

    # print the Top 5 recipes
    print("\nTop 5 SBERT Recommendations:")
    for rank, idx in enumerate(top_indices, start=1):
        name = df.iloc[idx]["Name"]
        ing = df.iloc[idx]["Ingredients"]
        sim = scores[idx]
        print(f"[{rank}] {name} | sim: {sim:.4f} | {ing[:]}")

# print the time elapsed
end_time = time.perf_counter()
print("\n elapsed time: ", end_time - start_time, " seconds")
