from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
csv_path = DATASET_DIR / "recipeex001.csv"

df = pd.read_csv(csv_path)
df["Name"] = df["Name"].fillna("Unnamed Recipe").astype(str)
df["Ingredients"] = df["Ingredients"].fillna("").astype(str).str.lower()

tfidf_texts = df["Ingredients"].str.replace(",", " ", regex=False)

# Step 1: Vectorization (TF-IDF)
vectorizer_norm2 = TfidfVectorizer(stop_words="english", norm="l2")
vectorizer_norm1 = TfidfVectorizer(stop_words="english", norm="l1")
tfidf_points_norm2 = vectorizer_norm2.fit_transform(tfidf_texts)
tfidf_points_norm1 = vectorizer_norm1.fit_transform(tfidf_texts)

# Step 2: Fit k-NN models for Euclidean, Manhattan, and Cosine
top_k = 5
models = {
    "Euclidean": NearestNeighbors(n_neighbors=top_k, metric="euclidean").fit(tfidf_points_norm2),
    "Manhattan": NearestNeighbors(n_neighbors=top_k, metric="manhattan").fit(tfidf_points_norm1),
    "Cosine": NearestNeighbors(n_neighbors=top_k, metric="cosine").fit(tfidf_points_norm2),
}

# Same seasonings set as sbert_analysis.py for a fair comparison
SEASONINGS = {
    "salt", "pepper", "black pepper", "oil", "sunflower oil", "vegetable oil",
    "olive oil", "garlic", "ginger", "onion", "onions", "water", "sugar",
    "soy sauce", "vinegar", "turmeric", "cumin", "cumin seeds", "coriander",
    "chilli", "chili", "red chilli powder", "mustard", "mustard seeds", "cloves",
    "cinnamon", "cardamom", "curry leaves", "asafoetida", "butter", "ghee",
    "fish sauce", "sesame oil", "mustard oil", "bay leaf", "chili flakes",
    "turmeric powder", "coriander powder", "garam masala", "chili powder",
}

# Same 60 test queries as sbert_analysis.py
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

summary_rows = []

for metric_name, model in models.items():
    for category_name, query_list in QUERIES.items():
        acc_list = []
        prec_list = []
        rec_list = []
        f1_list = []
        speed_list = []
        extra_err_list = []
        bias_err_list = []

        for query in query_list:
            # 1. Run k-NN search and time it in milliseconds
            start = time.perf_counter()
            cleaned_query = query.replace(",", " ").lower()
            if metric_name == "Manhattan":
                query_point = vectorizer_norm1.transform([cleaned_query])
            else:
                query_point = vectorizer_norm2.transform([cleaned_query])
            distances, indices = model.kneighbors(query_point)
            top_indices = indices[0]
            elapsed_ms = (time.perf_counter() - start) * 1000
            speed_list.append(elapsed_ms)

            # 2. Split query into primary (P) and secondary/seasoning (S) ingredients
            user_items = [item.strip().lower() for item in query.split(",") if item.strip()]
            primary = []
            secondary = []
            for u in user_items:
                if u in SEASONINGS:
                    secondary.append(u)
                else:
                    primary.append(u)

            # 3. Check each of the top 5 recommended recipes
            relevant_count = 0
            valid_count = 0
            extra_errors = 0
            bias_errors = 0

            for idx in top_indices:
                recipe_text = df.iloc[idx]["Ingredients"]
                recipe_items = [item.strip() for item in recipe_text.split(",") if item.strip()]

                # Count missing ingredients (T)
                missing_count = 0
                for rec_ing in recipe_items:
                    found = False
                    for u in user_items:
                        if u in rec_ing:
                            found = True
                            break
                    if not found:
                        missing_count += 1

                # Check if primary (P) was dropped or secondary (S) matched
                dropped_primary = False
                for p in primary:
                    if p not in recipe_text:
                        dropped_primary = True
                        break

                matched_secondary = False
                for s in secondary:
                    if s in recipe_text:
                        matched_secondary = True
                        break

                # Accuracy: contains all user's key/primary ingredients
                if not dropped_primary:
                    relevant_count += 1

                # Precision/Valid: matches key ingredients AND missing ingredients T <= 4
                if not dropped_primary and missing_count <= 4:
                    valid_count += 1

                # Error 1: Extra Ingredient (T > 4)
                if missing_count > 4:
                    extra_errors += 1

                # Error 2: Ingredient Bias (matched S while dropping P)
                if matched_secondary and dropped_primary:
                    bias_errors += 1

            # 4. Count total valid recipes in the entire dataset for Recall
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

            # 5. Compute metrics for this query
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
            "Metric": metric_name,
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
print("\n=== k-NN 60-QUERY ANALYSIS RESULTS ===")
print(results_df.to_string(index=False))
