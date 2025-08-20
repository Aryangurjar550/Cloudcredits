# Stock Price Prediction with LSTM (PyTorch, NSE-Ready)

This is the **PyTorch version** of the stock price prediction project, ready for Indian markets (NSE).

## Examples
```bash
# Reliance
python -m src.train --ticker RELIANCE.NS --start 2015-01-01 --end 2025-08-01 --epochs 20
python -m src.evaluate --ticker RELIANCE.NS --start 2015-01-01 --end 2025-08-01

# NIFTY 50
python -m src.train --ticker ^NSEI --start 2015-01-01 --end 2025-08-01 --epochs 20
python -m src.evaluate --ticker ^NSEI --start 2015-01-01 --end 2025-08-01
```

Outputs:
- `outputs/metrics.json`
- `outputs/pred_vs_actual.png`
- `models/model.pt` (PyTorch model)
- `models/scaler.pkl`

Dependencies are listed in `requirements.txt`.
