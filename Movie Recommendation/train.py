import argparse
import os
from surprise import Dataset, SVD, KNNBaseline
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate
from surprise import Reader
import joblib

def get_algo(name: str, n_factors: int, epochs: int):
    name = name.lower()
    if name == "svd":
        return SVD(n_factors=n_factors, n_epochs=epochs, random_state=42, verbose=True)
    elif name == "knn":
        # Baseline-enabled user-based CF
        sim_options = {"name": "pearson_baseline", "user_based": True}
        return KNNBaseline(sim_options=sim_options, verbose=True)
    else:
        raise ValueError("Unknown algo. Use 'svd' or 'knn'.")

def top_n_for_user(algo, trainset, user_raw_id: str, topk: int = 10):
    # If the user id is numeric (like 196), MovieLens raw ids are also strings
    user_raw_id = str(user_raw_id)
    # Build predictions for items the user hasn't rated
    anti_testset = trainset.build_anti_testset(fill=None)  # (uid, iid, fill)
    # Filter only this user's anti-test entries for speed
    anti_user = [(user_raw_id, iid, 0.0) for (uid, iid, _) in anti_testset if uid == user_raw_id]
    # If anti_user is empty, fallback to the full anti_testset but we'll filter post-prediction
    if not anti_user:
        preds = algo.test(anti_testset)
        preds = [p for p in preds if p.uid == user_raw_id]
    else:
        preds = algo.test(anti_user)
    # Sort by estimated rating descending
    preds.sort(key=lambda x: x.est, reverse=True)
    return preds[:topk]

def main():
    parser = argparse.ArgumentParser(description="MovieLens Recommender (Collaborative Filtering)")
    parser.add_argument("--algo", type=str, default="svd", choices=["svd", "knn"], help="Algorithm to use")
    parser.add_argument("--n-factors", type=int, default=100, help="Latent factors for SVD")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for SVD")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size for hold-out split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--user-id", type=str, default="196", help="User id for recommendations (MovieLens 100K example)")
    parser.add_argument("--topk", type=int, default=10, help="Number of recommendations to show")
    parser.add_argument("--do-cv", action="store_true", help="Also run 3-fold cross-validation (RMSE/MAE)")
    args = parser.parse_args()

    # Load MovieLens 100K (built-in). To use custom csv:
    # ratings = pd.read_csv("ratings.csv"); reader = Reader(rating_scale=(0.5,5)); data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
    data = Dataset.load_builtin("ml-100k")

    # Optional cross-validation
    if args.do_cv:
        print("Running 3-fold cross-validation...")
        cv_results = cross_validate(get_algo(args.algo, args.n_factors, args.epochs), data, measures=["RMSE", "MAE"], cv=3, verbose=True)
        print("CV Results:", cv_results)

    # Train / test split
    trainset, testset = train_test_split(data, test_size=args.test_size, random_state=args.random_state)

    # Fit algorithm
    algo = get_algo(args.algo, args.n_factors, args.epochs)
    algo.fit(trainset)

    # Evaluate on test
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    print(f"Test RMSE: {rmse:.4f}")

    # Ensure models dir
    os.makedirs("models", exist_ok=True)
    # Save model + trainset for later inference
    joblib.dump({"algo": algo, "trainset": trainset}, "models/model.pkl")
    print("Saved model to models/model.pkl")

    # Generate top-N recommendations for a user
    try:
        top_preds = top_n_for_user(algo, trainset, args.user_id, args.topk)
        print(f"\nTop-{args.topk} recommendations for user {args.user_id}:")
        for i, p in enumerate(top_preds, 1):
            print(f"{i:2d}. item_id={p.iid} | est_rating={p.est:.3f}")
    except Exception as e:
        print("Could not generate recommendations:", e)

if __name__ == "__main__":
    main()
