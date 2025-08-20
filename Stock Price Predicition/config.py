from dataclasses import dataclass

@dataclass
class Config:
    lookback: int = 60
    test_size: float = 0.2
    epochs: int = 20
    batch_size: int = 32
    lstm_units: int = 64
    dense_units: int = 32
    dropout: float = 0.2
    learning_rate: float = 1e-3
    target_column: str = "Close"
