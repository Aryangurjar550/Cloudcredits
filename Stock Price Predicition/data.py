from pathlib import Path
import pandas as pd
import yfinance as yf

def download_prices(ticker: str, start: str, end: str, cache_dir: str = "data") -> pd.DataFrame:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) / f"{ticker}_{start}_{end}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index().rename(columns={"Date": "Date"})
    df.to_csv(cache_path, index=False)
    df = df.set_index("Date")
    return df

def split_train_test(df: pd.DataFrame, test_size: float = 0.2):
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    return train, test
