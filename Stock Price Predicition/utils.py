import numpy as np
import json
from pathlib import Path

def create_sequences(array, lookback):
    X, y = [], []
    for i in range(lookback, len(array)):
        X.append(array[i - lookback:i])
        y.append(array[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

def save_metrics(metrics: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
