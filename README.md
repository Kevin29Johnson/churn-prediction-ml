# 🔁 Churn Prediction Using Artificial Neural Networks (ANN)

A machine learning project that predicts customer churn for a bank using an Artificial Neural Network built with TensorFlow/Keras. The goal is to identify which customers are likely to leave the bank based on their demographic and account activity features.

---

## 📌 Problem Statement

Customer churn significantly impacts a company's revenue. Being able to **predict churn** allows businesses to implement **retention strategies** early on. This project tackles a **binary classification** problem: will a customer churn (`Exited = 1`) or stay (`Exited = 0`)?

---

## 🧠 Model Overview

This project uses an **Artificial Neural Network (ANN)** for classification. The pipeline includes:
- Data preprocessing
- Feature encoding
- Train-test split with scaling
- ANN model training with dropout and early stopping
- Evaluation using accuracy and confusion matrix

---

## 📊 Dataset

**Source:** Kaggle / IBM Sample Dataset  
**File:** `Churn Modeling.csv`

**Features:**
- `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- **Target:** `Exited` (0 = stayed, 1 = churned)

---

## ⚙️ Tech Stack

- **Language:** Python
- **Libraries:** TensorFlow/Keras, Pandas, NumPy, Seaborn, Scikit-learn, Matplotlib
- **Tools:** Google Colab

---

## 🔄 Workflow

1. **Exploratory Data Analysis**
   - Checked for class imbalance
   - No missing values found
2. **Data Preprocessing**
   - One-hot encoding for categorical variables
   - Feature scaling using `StandardScaler`
3. **Model Architecture**
   - 3 hidden layers with `ReLU` activation
   - Dropout layers to reduce overfitting
   - Output layer with `sigmoid` for binary prediction
4. **Training**
   - Optimizer: `Adam` with learning rate = 0.01
   - Loss: `binary_crossentropy`
   - Metric: `accuracy`
   - Early stopping applied based on validation loss
5. **Evaluation**
   - Confusion Matrix
   - Accuracy Score
   - Visualization of training vs validation performance

---

## 📈 Results

- **Test Accuracy:** ~85%  
- **Model Exported:** Saved as `.keras` file for future use

**Confusion Matrix:**

[[1552,  43],
 [220, 185]]
