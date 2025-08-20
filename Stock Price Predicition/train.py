import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data import download_prices
from model import LSTMModel
from utils import create_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

def main():
    parser = argparse.ArgumentParser(description="Train PyTorch LSTM for stock price prediction")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=Config.lookback)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--hidden-size", type=int, default=Config.lstm_units)
    parser.add_argument("--dense-units", type=int, default=Config.dense_units)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--lr", type=float, default=Config.learning_rate)
    parser.add_argument("--target", type=str, default=Config.target_column)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = download_prices(args.ticker, args.start, args.end)
    if args.target not in df.columns:
        raise ValueError(f"Target column {args.target} not found. Available: {list(df.columns)}")

    series = df[args.target].astype("float32").values
    split_idx = int(len(series) * (1 - Config.test_size))
    train_values = series[:split_idx]
    test_values = series[split_idx:]

    scaler = MinMaxScaler()
    scaler.fit(train_values.reshape(-1,1))
    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    train_scaled = scaler.transform(train_values.reshape(-1,1)).reshape(-1)
    test_scaled = scaler.transform(test_values.reshape(-1,1)).reshape(-1)

    X_train, y_train = create_sequences(train_scaled, args.lookback)
    X_test, y_test = create_sequences(np.concatenate([train_scaled[-args.lookback:], test_scaled]), args.lookback)

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)

    model = LSTMModel(input_size=1, hidden_size=args.hidden_size, dense_units=args.dense_units, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss={epoch_loss/len(train_loader):.6f}")

    torch.save(model.state_dict(), "models/model.pt")

    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test.to(device)).cpu().numpy().reshape(-1)
    y_test_np = y_test.numpy().reshape(-1)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).reshape(-1)
    y_true = scaler.inverse_transform(y_test_np.reshape(-1,1)).reshape(-1)

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
    plt.title(f"Prediction vs Actual — {args.ticker}")
    plt.savefig("outputs/pred_vs_actual.png", bbox_inches="tight")

    print(f"Done. MAE={mae:.4f}, RMSE={rmse:.4f}")

if __name__ == "__main__":
    main()
