# Credit-Card-Fraud-Detection-Using-ML

This project focuses on detecting fraudulent credit card transactions using Logistic Regression, a simple yet effective machine learning classification algorithm. The dataset used is highly imbalanced, and special care has been taken to handle this issue during preprocessing.

## 📌 Project Overview

Credit card fraud is a growing concern in the financial industry. The objective of this project is to build a model that can accurately detect fraudulent transactions based on a set of anonymized features derived from real-world credit card data.

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn (for visualization)
- Jupyter Notebook

## 📂 Dataset

The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by European cardholders in September 2013.

- **Total records:** 284,807
- **Fraudulent transactions:** 492 (~0.17%)

## 🧹 Data Preprocessing

- Handled class imbalance using techniques like SMOTE and undersampling
- Scaled features using StandardScaler
- Removed outliers where appropriate
- Split data into training and test sets

## 🤖 Model: Logistic Regression

Logistic Regression was chosen for its simplicity, interpretability, and strong performance as a baseline model.

### Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

## 📈 Results

- The model performed well, especially in terms of **precision and recall**, which are crucial for fraud detection.
- Demonstrated strong baseline results and interpretability.

## 🚀 Future Work

- Experiment with advanced models (e.g., Random Forest, XGBoost, Neural Networks)
- Deploy model for real-time fraud detection
- Optimize feature engineering and hyperparameters

## 📝 Conclusion

This project shows that Logistic Regression, when paired with proper preprocessing and evaluation, can effectively detect fraudulent transactions and serve as a strong starting point in the fight against financial fraud.
