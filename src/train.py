import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
from logistic_model import LogisticRegressionScratch
from data_loader import load_data

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

FILE_PATH = "/home/kp17/Code/Projects/stock-price-classifier/data/processed/HDFCBANK.NS.csv"
LEARNING_RATES = [0.001, 0.002, 0.003]
EPOCHS = 1000

x_train, x_test, y_train, y_test = load_data(FILE_PATH)

final_accuracies = []

os.makedirs("results", exist_ok=True)

for lr in LEARNING_RATES:
    print(f"\n Training with learning rate: {lr}")
    
    model = LogisticRegressionScratch(learning_rate=lr, epochs=EPOCHS)
    model.fit(x_train, y_train, x_test, y_test)

    acc = np.mean(model.predict(x_test) == y_test)
    final_accuracies.append(acc)

    tag = str(lr).replace(".", "p")

    print(f"[DEBUG] LR={lr} | Epochs tracked: {len(model.losses)} | Final acc: {acc:.4f}")

    # === Loss Plot ===
    loss_data = model.losses
    if len(loss_data) > 10:
        loss_data = smooth(loss_data)

    epochs = list(range(len(loss_data)))
    plt.plot(epochs, loss_data, linewidth=2.5)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Loss vs Epochs (LR={lr})")
    plt.xticks(np.arange(0, len(epochs) + 1, max(1, len(epochs) // 10)))
    plt.ylim(min(loss_data) - 0.01, max(loss_data) + 0.01)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.savefig(f"stock-price-classifier/results/loss_vs_epochs_lr_{tag}.png")
    plt.clf()

    # === Accuracy Plot ===
    train_acc = model.train_accuracies
    test_acc = model.test_accuracies
    if len(train_acc) > 10:
        train_acc = smooth(train_acc)
    if len(test_acc) > 10:
        test_acc = smooth(test_acc)

    epochs = list(range(len(train_acc)))
    plt.plot(epochs, train_acc, label="Train Acc", linewidth=2.5)
    plt.plot(epochs, test_acc, label="Test Acc", linewidth=2.5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Epochs (LR={lr})")
    plt.xticks(np.arange(0, len(epochs) + 1, max(1, len(epochs) // 10)))
    plt.ylim(0.4, 1.0)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.savefig(f"stock-price-classifier/results/accuracy_vs_epochs_lr_{tag}.png")
    plt.clf()

plt.plot(LEARNING_RATES, final_accuracies, marker='o', linewidth=2.5)
plt.xlabel("Learning Rate")
plt.ylabel("Final Test Accuracy")
plt.title("Learning Rate vs Test Accuracy")
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.savefig("stock-price-classifier/results/lr_vs_accuracy.png")
plt.clf()

best_index = np.argmax(final_accuracies)
best_lr = LEARNING_RATES[best_index]

print(f"\n Best LR: {best_lr} | Accuracy: {final_accuracies[best_index]:.4f}")

best_model = LogisticRegressionScratch(learning_rate=best_lr, epochs=EPOCHS)
best_model.fit(x_train, y_train, x_test, y_test)

os.makedirs("models", exist_ok=True)
np.save("stock-price-classifier/models/theta.npy", best_model.theta)
print(" Best model weights saved to models/theta.npy")
