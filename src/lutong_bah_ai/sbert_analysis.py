from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import time

# same explanation from sbert.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
csv_path = DATASET_DIR / "recipeex001.csv"

df = pd.read_csv(csv_path)
df["Name"] = df["Name"].fillna("Unnamed Recipe").astype(str)
df["Ingredients"] = df["Ingredients"].fillna("").astype(str).str.lower()
recipe_texts = df["Ingredients"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

recipe_embeddings = model.encode(
    recipe_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

# secondary ingredients (seasonings) used to detect ingredient bias
# check last page of the proposal
SEASONINGS = {
    "salt", "pepper", "black pepper", "oil", "sunflower oil", "vegetable oil",
    "olive oil", "garlic", "ginger", "onion", "onions", "water", "sugar",
    "soy sauce", "vinegar", "turmeric", "cumin", "cumin seeds", "coriander",
    "chilli", "chili", "red chilli powder", "mustard", "mustard seeds", "cloves",
    "cinnamon", "cardamom", "curry leaves", "asafoetida", "butter", "ghee",
    "fish sauce", "sesame oil", "mustard oil", "bay leaf", "chili flakes",
    "turmeric powder", "coriander powder", "garam masala", "chili powder",
}

# 60 test queries (20 per category, check page 7 of the proposal)
QUERIES = {
    "Category 1 (Sparse)": [
        "chicken, broccoli", "beef, potatoes", "pork, cabbage", "egg, tomato",
        "fish, ginger", "tofu, spinach", "paneer, butter", "shrimp, garlic",
        "chicken, rice", "potato, peas", "mushroom, cream", "salmon, lemon",
        "pasta, tomato", "mutton, onion", "lentils, rice", "eggplant, garlic",
        "cauliflower, potato", "beans, pork", "carrot, chicken", "squash, coconut milk",
    ],
    "Category 2 (Balanced)": [
        "chicken, rice, garlic, soy sauce, tofu",
        "beef, onion, soy sauce, black pepper, rice",
        "pork, vinegar, soy sauce, garlic, bay leaf",
        "fish, tomato, onion, ginger, tamarind",
        "paneer, tomato, onion, butter, garam masala",
        "chicken, coconut milk, curry powder, ginger, onion",
        "shrimp, garlic, butter, lemon, parsley",
        "pasta, tomato, olive oil, garlic, basil",
        "chicken, egg, onion, garlic, soy sauce",
        "lentils, turmeric, cumin seeds, garlic, tomato",
        "chickpeas, onion, tomato, ginger, garam masala",
        "tofu, soy sauce, garlic, ginger, sesame oil",
        "beef, potatoes, carrots, tomato paste, garlic",
        "pork, cabbage, onion, garlic, black pepper",
        "chicken, potatoes, carrots, curry powder, onion",
        "fish, coconut milk, ginger, green chilli, garlic",
        "egg, onion, tomato, green chilli, coriander",
        "noodles, chicken, cabbage, soy sauce, garlic",
        "mutton, yogurt, onion, ginger, garlic",
        "shrimp, tamarind, tomato, onion, radish",
    ],
    "Category 3 (Side-Heavy)": [
        "fish, garlic, soy sauce, pepper, onions, ginger",
        "chicken, salt, pepper, garlic, ginger, oil, turmeric",
        "pork, garlic, onion, vinegar, soy sauce, black pepper, bay leaf, sugar",
        "beef, garlic, onion, ginger, soy sauce, pepper, salt",
        "tofu, soy sauce, sesame oil, garlic, ginger, chili, scallion",
        "paneer, cumin seeds, turmeric, coriander powder, garam masala, chili powder, salt, oil",
        "shrimp, butter, garlic, pepper, salt, lemon, parsley, chili flakes",
        "chicken, cumin seeds, coriander powder, turmeric powder, garam masala, ginger, garlic, salt",
        "fish, turmeric, red chilli powder, coriander powder, mustard oil, salt, garlic",
        "egg, salt, black pepper, butter, garlic, onion, green chilli",
        "rice, garlic, onion, salt, black pepper, butter",
        "squash, garlic, onion, ginger, black pepper, fish sauce, coconut milk",
        "eggplant, garlic, onion, vinegar, soy sauce, black pepper, sugar",
        "cauliflower, cumin seeds, turmeric powder, red chilli powder, ginger, garlic, salt",
        "potatoes, cumin seeds, mustard seeds, curry leaves, turmeric, salt, oil",
        "mutton, cinnamon, cloves, cardamom, ginger, garlic, onion, salt",
        "lentils, cumin seeds, mustard seeds, garlic, dry red chilli, curry leaves, asafoetida, ghee",
        "cabbage, mustard seeds, curry leaves, green chilli, coconut, salt, oil",
        "spinach, garlic, cumin seeds, green chilli, butter, cream, salt",
        "chicken, mustard oil, ginger, garlic, red chilli powder, garam masala, yogurt, salt",
    ],
}

top_k = 5
summary_rows = []

# compute the metrics + error per category
for category_name, query_list in QUERIES.items():
    # ito yung mga metrics natin + errors
    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    speed_list = []
    extra_err_list = []
    bias_err_list = []

    # for each query in each category (20 queries per category)
    for query in query_list:
        # start timer
        start = time.perf_counter()
        # embed the query
        query_vector = model.encode([query.lower()], normalize_embeddings=True)[0]
        # do cosine similarity
        scores = np.dot(recipe_embeddings, query_vector)
        # sort highest to lowest
        top_indices = np.argsort(scores)[::-1][:top_k]
        # compute how long it took
        elapsed_ms = (time.perf_counter() - start) * 1000
        speed_list.append(elapsed_ms)

        # start of the analysis part!!!
        # split query into primary (P) and secondary/seasoning (S) ingredients
        # used to compute ingredient bias
        user_items = [item.strip().lower() for item in query.split(",") if item.strip()]
        primary = []
        secondary = []
        for u in user_items:
            if u in SEASONINGS:
                secondary.append(u)
            else:
                primary.append(u)

        # check each of the top 5 recommended recipes
        relevant_count = 0
        valid_count = 0
        extra_errors = 0
        bias_errors = 0
        for idx in top_indices:
            # get the actual ingredients of this current recommendation for this category
            recipe_text = df.iloc[idx]["Ingredients"]
            recipe_items = [item.strip() for item in recipe_text.split(",") if item.strip()]

            # count how many ingredients are missing from the recommendation
            # given the query
            missing_count = 0
            for rec_ing in recipe_items:
                found = False
                for u in user_items:
                    if u in rec_ing:
                        found = True
                        break
                if not found:
                    missing_count += 1

            # used for ingredient bias
            # Check if primary (P) was dropped
            # meaning at least 1 primary is not present
            dropped_primary = False
            for p in primary:
                if p not in recipe_text:
                    dropped_primary = True
                    break
            # check if secondary (S) matched
            # meaning there is at least 1 secondary
            matched_secondary = False
            for s in secondary:
                if s in recipe_text:
                    matched_secondary = True
                    break

            # for each recommendation in this category
            # accuracy: current recommendation contains all user's key/primary ingredients
            if not dropped_primary:
                relevant_count += 1

            # precision: matches key ingredients AND missing ingredients T <= 4
            if not dropped_primary and missing_count <= 4:
                valid_count += 1

            # error 1: Extra Ingredient (T > 4)
            if missing_count > 4:
                extra_errors += 1

            # error 2: Ingredient Bias (matched S while dropping P)
            if matched_secondary and dropped_primary:
                bias_errors += 1

        # count total valid recipes in the entire dataset for Recall
        total_valid_in_dataset = 0
        for text in df["Ingredients"]:
            has_all_primary = True
            for p in primary:
                if p not in text:
                    has_all_primary = False
                    break
            if has_all_primary:
                total_valid_in_dataset += 1

        if total_valid_in_dataset == 0:
            total_valid_in_dataset = 1

        # compute metrics for this query
        accuracy = relevant_count / top_k
        precision = valid_count / top_k
        recall = min(1.0, valid_count / total_valid_in_dataset)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        acc_list.append(accuracy)
        prec_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)
        extra_err_list.append((extra_errors / top_k) * 100)
        bias_err_list.append((bias_errors / top_k) * 100)

    summary_rows.append({
        "Category": category_name,
        "Accuracy": round(float(np.mean(acc_list)), 4),
        "Precision": round(float(np.mean(prec_list)), 4),
        "Recall": round(float(np.mean(rec_list)), 4),
        "F1-Score": round(float(np.mean(f1_list)), 4),
        "Speed (ms)": round(float(np.mean(speed_list)), 2),
        "Extra Ing (%)": round(float(np.mean(extra_err_list)), 1),
        "Ing Bias (%)": round(float(np.mean(bias_err_list)), 1),
    })

results_df = pd.DataFrame(summary_rows)
print("\n=== SBERT 60-QUERY ANALYSIS RESULTS ===")
print(results_df.to_string(index=False))
