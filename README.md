![Cover page Lutong-Bah-AI](docs/cover.jpg)
An ingredient-based recipe recommendation system built with **k-Nearest Neighbors (k-NN + TF-IDF)** and **Sentence-BERT (SBERT)**. Given a list of available pantry ingredients, both models recommend the top 5 matching recipes from a dataset of 5,938 recipes.

---

## Project Structure

```text
Lutong-Bah-AI/
├── dataset/
│   └── recipeex001.csv          # Recipe dataset (5,938 recipes)
├── src/
│   └── lutong_bah_ai/
│       ├── __init__.py
│       ├── kNN.py               # Model 1: TF-IDF + k-NN (Euclidean, Manhattan, Cosine)
│       └── sbert.py             # Model 2: Sentence-BERT (all-MiniLM-L6-v2 + Cosine)
├── pyproject.toml               # Project metadata & dependencies
├── uv.lock                      # Locked dependency versions
└── README.md
```

---

## Prerequisites

This project uses [`uv`](https://docs.astral.sh/uv/) for Python and package management.

### Install `uv`

- **macOS / Linux:**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

---

## Setup

1. **Clone the repository and enter the project folder:**

   ```bash
   git clone <your-repo-url>
   cd Lutong-Bah-AI
   ```

2. **Install all dependencies (`uv` will set up Python and `.venv` automatically):**
   ```bash
   uv sync
   ```

> **Note:** If `sentence-transformers` is not yet in `pyproject.toml`, install it with:
>
> ```bash
> uv add sentence-transformers
> ```

---

## How to Run the Models

Always run commands from the root `Lutong-Bah-AI` folder.

### Option 1: Run k-Nearest Neighbors (`kNN.py`)

Compares **Euclidean**, **Manhattan**, and **Cosine** distance using TF-IDF vectors:

```bash
uv run src/lutong_bah_ai/kNN.py
```

When prompted, enter comma-separated ingredients:

```text
Enter ingredients (e.g., chicken, garlic, salt): chicken, garlic, soy sauce
```

---

### Option 2: Run Sentence-BERT (`sbert.py`)

Uses the pre-trained `all-MiniLM-L6-v2` model to encode ingredients into 384-dimensional semantic vectors and rank recipes using Cosine Similarity:

```bash
uv run src/lutong_bah_ai/sbert.py
```

When prompted, enter comma-separated ingredients:

```text
Enter ingredients (e.g., chicken, garlic, soy sauce): chicken, garlic, soy sauce
```

> **First-run note for SBERT:** The very first time you run `sbert.py`, it will download the `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face and encode all 5,938 recipes (~10–20 seconds depending on your PC).

---

## Adding New Packages

If you need another library, install it with `uv add` so `pyproject.toml` and `uv.lock` stay updated for your teammates:

```bash
uv add <package-name>
```
