import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loader import load_data
from regression_model import LinearRegressionScratch

# === Load Data ===
FILE_PATH = "data/processed/HDFCBANK.NS.csv"
x_train, x_test, y_train, y_test = load_data(FILE_PATH)

# === Train Scratch Model ===
scratch_model = LinearRegressionScratch(learning_rate=0.001, epochs=1000)
scratch_model.fit(x_train, y_train)
y_pred_scratch = scratch_model.predict(x_test)

# === Train Sklearn Model ===
sk_model = LinearRegression()
sk_model.fit(x_train, y_train)
y_pred_sklearn = sk_model.predict(x_test)

# === Print Evaluation ===
print("🧠 Scratch Linear Regression")
print("---------------------------")
print("MSE :", mean_squared_error(y_test, y_pred_scratch))
print("MAE :", mean_absolute_error(y_test, y_pred_scratch))
print("R²  :", r2_score(y_test, y_pred_scratch))

print("\n📦 Sklearn Linear Regression")
print("----------------------------")
print("MSE :", mean_squared_error(y_test, y_pred_sklearn))
print("MAE :", mean_absolute_error(y_test, y_pred_sklearn))
print("R²  :", r2_score(y_test, y_pred_sklearn))

# === Plot Comparison ===
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual Returns", linewidth=2)
plt.plot(y_pred_scratch, label="Scratch Model", linestyle="--", linewidth=2)
plt.plot(y_pred_sklearn, label="Sklearn Model", linestyle="--", linewidth=2)
plt.title("Actual vs Predicted Returns Over Time")
plt.xlabel("Test Sample Index")
plt.ylabel("Return")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Save to results
import os
results_path = os.path.join(os.path.dirname(__file__), "../results")

os.makedirs(results_path, exist_ok=True)
plt.savefig("results/compare_scratch_vs_sklearn.png")
plt.show()
