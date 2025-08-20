import argparse
from pathlib import Path
import numpy as np
import torch
import pickle

from config import Config
from data import download_prices
from model import LSTMModel
from utils import create_sequences
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved PyTorch LSTM model")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=Config.lookback)
    parser.add_argument("--target", type=str, default=Config.target_column)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = download_prices(args.ticker, args.start, args.end)
    series = df[args.target].astype("float32").values

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    series_scaled = scaler.transform(series.reshape(-1,1)).reshape(-1)

    X, y = create_sequences(series_scaled, args.lookback)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)

    model = LSTMModel()
    model.load_state_dict(torch.load("models/model.pt", map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds_scaled = model(X).cpu().numpy().reshape(-1)

    y_np = y.cpu().numpy().reshape(-1)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).reshape(-1)
    y_true = scaler.inverse_transform(y_np.reshape(-1,1)).reshape(-1)

    mae = float(mean_absolute_error(y_true, preds))
    rmse = float(mean_squared_error(y_true, preds, squared=False))

    from utils import save_metrics
    save_metrics({"MAE": mae, "RMSE": rmse}, "outputs/metrics.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path("outputs").mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title(f"Prediction vs Actual — {args.ticker} (PyTorch)")
    plt.savefig("outputs/pred_vs_actual.png", bbox_inches="tight")

    print("Evaluation complete: MAE=%.4f, RMSE=%.4f" % (mae, rmse))

if __name__ == "__main__":
    main()
