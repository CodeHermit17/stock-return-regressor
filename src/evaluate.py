import numpy as np
import os
import pandas as pd
from regression_model import LinearRegressionScratch
from data_loader import load_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FILE_PATH = "/home/kp17/Code/Projects/stock-return-regressor/data/processed/ITC.NS.csv"
MODEL_PATH = "/home/kp17/Code/Projects/stock-return-regressor/models/theta.npy"
_, x_test, _, y_test = load_data(FILE_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}. Run train.py first.")

theta = np.load(MODEL_PATH)

model = LinearRegressionScratch(learning_rate=0.001, epochs=0)
model.theta = theta

# === Predict on Test Set ===
y_pred = model.predict(x_test)

# === Evaluate Performance ===
print("Evaluation Metrics:")
print("MSE :", mean_squared_error(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("R²  :", r2_score(y_test, y_pred))

# === Save Predictions for Inspection ===
df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
os.makedirs("/home/kp17/Code/Projects/stock-return-regressor/results", exist_ok=True)
df.to_csv("/home/kp17/Code/Projects/stock-return-regressor/results/evaluation_predictions.csv", index=False)
print("\nPredictions saved to results/evaluation_predictions.csv")
