import pandas as pd
import numpy as np

def load_data(file_path: str):

    np.set_printoptions(formatter={'float_kind': '{:.2f}'.format})
    
    df = pd.read_csv(file_path)

    feature_cols = ['SMA10', 'SMA20', 'Momentum', 'Daily_Return', 'SMA_diff', 'Range']
    x_main = df[feature_cols].to_numpy()
    y_main = df['Target'].to_numpy()

    valid_rows = ~(np.isnan(x_main).any(axis=1))
    x_main = x_main[valid_rows]
    y_main = y_main[valid_rows]

    x_main = x_main[:-1]
    y_main = y_main[:-1]

    split_idx = int(len(x_main) * 0.8)
    x_train = x_main[:split_idx]
    x_test = x_main[split_idx:]
    y_train = y_main[:split_idx]
    y_test = y_main[split_idx:]

    mean_train = np.mean(x_train, axis=0)
    std_train = np.std(x_train, axis=0)

    x_train_scaled = (x_train - mean_train) / std_train
    x_test_scaled = (x_test - mean_train) / std_train  

    assert x_train_scaled.shape[0] == y_train.shape[0], "Train shapes mismatch"
    assert x_test_scaled.shape[0] == y_test.shape[0], "Test shapes mismatch"

    return x_train_scaled, x_test_scaled, y_train, y_test

