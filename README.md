# 🧠 Census Income Prediction: KNN vs. Neural Networks

A comparative machine learning project that predicts whether an individual's income exceeds $50,000/year based on census data. This project implements, evaluates, and contrasts a traditional baseline model against a deep learning approach.

---

## 📊 Project Overview

This repository contains a full-stack data science workflow analyzing demographic census data. The goal is to evaluate the performance trade-offs between a lightweight **K-Nearest Neighbors (KNN)** model and a multi-layer **Sequential Neural Network**.

### Key Findings

* **Performance Tie:** Both models achieved a virtual statistical tie on the unseen test set, with the Neural Network hitting **85.48% accuracy** and the KNN model matching it closely at **85.44%**.
* **Complexity Trade-off:** The added engineering overhead, tuning requirements, and training time of the deep learning model were not justified given the identical performance returns.

---

## 🛠️ Tech Stack & Architecture

### Core Dependencies

* **Data Processing:** `pandas`, `numpy`, `scikit-learn`
* **Deep Learning Framework:** `tensorflow.keras`
* **Visualization:** `matplotlib`

### Neural Network Architecture

* **Input Layer:** Handles the preprocessed tabular feature space.
* **Hidden Layers:** Two dense layers (32 and 16 units) utilizing `relu` activations for fast, stable gradient propagation.
* **Output Layer:** A single dense unit with a `sigmoid` activation function to output binary classification probabilities.
* **Optimization:** Trained using Stochastic Gradient Descent (SGD) with a baseline learning rate of `0.01` over `30` epochs.

---

## 🚀 How to Load and Use the Models

The repository includes the fully trained weights saved directly from the notebook pipeline.

```python
import pickle
from tensorflow.keras.models import load_model

# 1. Load the data scaler (if processing raw inputs)
scaler = pickle.load(open('scaler.pkl', 'rb'))

# 2. Load the trained Neural Network
nn_model = load_model('nn_model.h5')

# 3. Predict probabilities on new data
# predictions = nn_model.predict(scaled_data)

```

---

## 📈 Model Performance Comparison

| Evaluation Metric | KNN Model Baseline | Sequential Neural Network |
| --- | --- | --- |
| **Test Accuracy** | 85.44% | **85.48%** |
| **Test F1 Score** | **66.99%** | 66.55% |

---

## 🔮 Future Enhancements

If expanding this repository further, next steps include:

1. **Handling Class Imbalance:** Implementing **SMOTE** (Synthetic Minority Over-sampling Technique) to boost the minority income class F1 scores.
2. **Hyperparameter Optimization:** Swapping out SGD for the `Adam` optimizer and running grid searches on layer configurations.
3. **Feature Engineering:** Building interaction terms between education levels and age brackets to reveal deeper non-linear boundaries.
