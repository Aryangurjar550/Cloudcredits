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
       
        sim_options = {"name": "pearson_baseline", "user_based": True}
        return KNNBaseline(sim_options=sim_options, verbose=True)
    else:
        raise ValueError("Unknown algo. Use 'svd' or 'knn'.")

def top_n_for_user(algo, trainset, user_raw_id: str, topk: int = 10):
   
    user_raw_id = str(user_raw_id)
  
    anti_testset = trainset.build_anti_testset(fill=None) 
   
    anti_user = [(user_raw_id, iid, 0.0) for (uid, iid, _) in anti_testset if uid == user_raw_id]
  
    if not anti_user:
        preds = algo.test(anti_testset)
        preds = [p for p in preds if p.uid == user_raw_id]
    else:
        preds = algo.test(anti_user)
   
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


    data = Dataset.load_builtin("ml-100k")


    if args.do_cv:
        print("Running 3-fold cross-validation...")
        cv_results = cross_validate(get_algo(args.algo, args.n_factors, args.epochs), data, measures=["RMSE", "MAE"], cv=3, verbose=True)
        print("CV Results:", cv_results)

   
    trainset, testset = train_test_split(data, test_size=args.test_size, random_state=args.random_state)


    algo = get_algo(args.algo, args.n_factors, args.epochs)
    algo.fit(trainset)

   
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    print(f"Test RMSE: {rmse:.4f}")

  
    os.makedirs("models", exist_ok=True)

    joblib.dump({"algo": algo, "trainset": trainset}, "models/model.pkl")
    print("Saved model to models/model.pkl")

 
    try:
        top_preds = top_n_for_user(algo, trainset, args.user_id, args.topk)
        print(f"\nTop-{args.topk} recommendations for user {args.user_id}:")
        for i, p in enumerate(top_preds, 1):
            print(f"{i:2d}. item_id={p.iid} | est_rating={p.est:.3f}")
    except Exception as e:
        print("Could not generate recommendations:", e)

if __name__ == "__main__":
    main()
