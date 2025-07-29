# 📉 Stock Return Regressor

This project aims to **predict the next-day percentage return of a stock** using a **linear regression model implemented from scratch in Python**. It builds directly on the earlier [`stock-price-classifier`](https://github.com/CodeHermit17/stock-price-classifier) project and is part of a broader roadmap focused on **machine learning + finance + GSoC internship prep**.

---

## 🚀 Project Goals

- Predict **numeric stock returns** (not binary up/down)
- Implement **Linear Regression from scratch** using NumPy
- Visualize loss curves and compare against sklearn model
- Build reproducible, modular pipeline for future hybrid models (e.g. sentiment + price)

---

## 🛠️ Tech Stack

- **Python** (core logic)
- **NumPy** (math/gradients)
- **Pandas** (data processing)
- **Matplotlib** (loss/visualization)
- **scikit-learn** (evaluation comparison only)

---

## 📊 Results Summary

- ✅ **MSE**: `0.00010378`
- ✅ **MAE**: `0.00756`
- ✅ **R² Score**: `0.01185`
- 🔁 Scratch model performance is **consistent with sklearn**
- 📉 Predicts daily returns with **low bias**, but limited variance capture (expected)

---

## 📁 Project Structure

stock-return-regressor/
├── data/
│ ├── raw/ # Original stock CSVs (e.g., HDFCBANK.NS.csv)
│ └── processed/ # Cleaned data with SMA, RSI, and returns
├── models/
│ └── theta.npy # Saved weights for scratch model
├── results/
│ ├── loss_vs_epochs_lr_0p001.png
│ └── compare_scratch_vs_sklearn.png
├── src/
│ ├── regression_model.py # Linear Regression (Scratch)
│ ├── prepare_data.py # Adds SMA, RSI, returns
│ ├── data_loader.py # Loads + splits processed data
│ ├── train.py # Trains model and logs loss
│ └── evaluate.py # Compares scratch vs sklearn
├── requirements.txt
└── README.md

---

## 🧠 Core Implementation

- ✅ `regression_model.py`: Manual implementation of Linear Regression (bias, loss, gradients)
- ✅ `train.py`: Trains model using gradient descent, logs loss
- ✅ `evaluate.py`: Compares predictions of scratch model vs sklearn
- ✅ `prepare_data.py`: Adds RSI, SMA, and return features
- ✅ `data_loader.py`: Handles data loading/splitting

---

## 📈 Sample Outputs

Saved in the `/results/` folder:

- `loss_vs_epochs_lr_0p001.png` – Loss vs Epochs for scratch model
- `compare_scratch_vs_sklearn.png` – Line plot comparing predictions from scratch vs sklearn models

---

## ✅ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/CodeHermit17/stock-return-regressor.git
cd stock-return-regressor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train scratch model
python src/train.py

# 4. Evaluate and compare with sklearn
python src/evaluate.py

