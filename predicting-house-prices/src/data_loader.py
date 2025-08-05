from sklearn.datasets import load_boston
import pandas as pd

def load_data():
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data['PRICE'] = boston.target
    return data
