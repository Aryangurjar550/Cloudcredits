import argparse
import os
import joblib

def main():
    parser = argparse.ArgumentParser(description="Generate recommendations for a user using a saved model.")
    parser.add_argument("--user-id", type=str, default="196", help="User id (raw id as in MovieLens)")
    parser.add_argument("--topk", type=int, default=10, help="Number of recommendations to show")
    parser.add_argument("--model-path", type=str, default="models/model.pkl", help="Path to saved model")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}. Run src/train.py first.")

    bundle = joblib.load(args.model_path)
    algo = bundle["algo"]
    trainset = bundle["trainset"]

    # Build anti-testset and filter to this user
    anti = trainset.build_anti_testset(fill=None)
    user_raw_id = str(args.user_id)
    preds = [p for p in algo.test(anti) if p.uid == user_raw_id]
    preds.sort(key=lambda x: x.est, reverse=True)

    print(f"Top-{args.topk} recommendations for user {args.user_id}:")
    for i, p in enumerate(preds[:args.topk], 1):
        print(f"{i:2d}. item_id={p.iid} | est_rating={p.est:.3f}")

if __name__ == "__main__":
    main()
