# Movie Recommendation System (MovieLens, Collaborative Filtering)

Recommend movies to users based on their past ratings using **Collaborative Filtering** on the **MovieLens** dataset, evaluated with **RMSE**.

## 🧠 Approach
- **Dataset:** MovieLens 100K (via `surprise` built-in loader).
- **Algorithm:** Matrix Factorization (SVD from `scikit-surprise`). You can also switch to KNN-Baseline.
- **Evaluation:** Root Mean Squared Error (RMSE) via a hold-out test set + optional cross-validation.

## 📦 Project Structure
```
movie-recommender/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── train.py
    └── infer.py
```

## 🚀 Quickstart
```bash
# 1) (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train & evaluate
python src/train.py --algo svd --n-factors 100 --epochs 20 --topk 10 --user-id 196
# The script prints RMSE on the test set and top-N recommendations for the user.

# 4) Later, generate recommendations using a saved model
python src/infer.py --user-id 196 --topk 10
```

> **Note**: The MovieLens 100K dataset is automatically downloaded by `surprise` on first run.
If you want to use a different dataset (e.g., ML-1M), adapt the loading logic (see comments in `train.py`).

## 📈 Outputs
- **Printed metrics:** RMSE on a held-out test set.
- **Saved artifacts:** `models/model.pkl` containing the trained algorithm and trainset mappings.

## ⚙️ CLI Options (train)
- `--algo` : `svd` (default) or `knn` (KNNBaseline).
- `--n-factors` : Latent factors for SVD (default: 100).
- `--epochs` : Training epochs for SVD (default: 20).
- `--user-id` : User id from MovieLens to generate top-N recommendations (default: 196).
- `--topk` : Number of recommendations to show (default: 10).
- `--test-size` : Test split size (default: 0.2).
- `--random-state` : RNG seed (default: 42).

## 🧪 Reproducibility
Set `--random-state` to fix train/test split and algorithm RNG.

## 🧾 License
MIT (or choose your own).

---

Made for quick GitHub deployment.
