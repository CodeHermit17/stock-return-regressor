import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
from regression_model import LinearRegressionScratch
from data_loader import load_data
from sklearn.metrics import mean_squared_error, r2_score

def smooth(data, weight=0.9):
    if len(data) <= 2:
        return data
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

FILE_PATH = "/home/kp17/Code/Projects/stock-return-regressor/data/processed/HDFCBANK.NS.csv"
LEARNING_RATES = [0.001, 0.002, 0.003]
EPOCHS = 1000

x_train, x_test, y_train, y_test = load_data(FILE_PATH)

final_mse = []

results_path = os.path.join(os.path.dirname(__file__), "../results")
models_path = os.path.join(os.path.dirname(__file__), "../models")

os.makedirs(results_path, exist_ok=True)

for lr in LEARNING_RATES:
    print(f"\n Training with learning rate: {lr}")
    
    model = LinearRegressionScratch(learning_rate=lr, epochs=EPOCHS)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    final_mse.append(mse)

    tag = str(lr).replace(".", "p")

    print(f"[DEBUG] LR={lr} | Epochs tracked: {len(model.losses)} | Final MSE: {mse:.6f}")

    # === Loss Plot ===
    loss_data = model.losses
    if len(loss_data) > 10:
        loss_data = smooth(loss_data)

    epochs = list(range(len(loss_data)))
    plt.plot(epochs, loss_data, linewidth=2.5)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MSE)")
    plt.title(f"Loss vs Epochs (LR={lr})")
    plt.xticks(np.arange(0, len(epochs) + 1, max(1, len(epochs) // 10)))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.savefig(os.path.join(results_path, f"loss_vs_epochs_lr_{tag}.png"))
    plt.clf()

    # === Prediction Plot ===
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.title(f"Predicted vs Actual (LR={lr})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(results_path, f"loss_vs_epochs_lr_{tag}.png"))
    plt.clf()

# === Learning Rate vs MSE ===
plt.plot(LEARNING_RATES, final_mse, marker='o', linewidth=2.5)
plt.xlabel("Learning Rate")
plt.ylabel("Final Test MSE")
plt.title("Learning Rate vs Test MSE")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(results_path, f"loss_vs_epochs_lr_{tag}.png"))
plt.clf()

# === Save Best Model ===
best_index = np.argmin(final_mse)
best_lr = LEARNING_RATES[best_index]

print(f"\n Best LR: {best_lr} | Test MSE: {final_mse[best_index]:.6f}")

best_model = LinearRegressionScratch(learning_rate=best_lr, epochs=EPOCHS)
best_model.fit(x_train, y_train)

os.makedirs(models_path, exist_ok=True)
np.save("/home/kp17/Code/Projects/stock-return-regressor/models/theta.npy", best_model.theta)
print(" Best model weights saved to models/theta.npy")
