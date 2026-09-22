# CodeAlpha Credit Scoring Model

A Machine Learning based Credit Scoring System developed as part of the CodeAlpha Machine Learning Internship.

## 📌 Project Overview

This project predicts the credit risk of a customer based on historical financial and personal information.

The system uses supervised machine learning classification algorithms and provides a Streamlit web interface for interactive predictions.

## 🎯 Objective

The main objective is to classify customers into:

- Good Credit
- Bad Credit

## 📊 Dataset

**Dataset:** Statlog German Credit Data  
**Source:** UCI Machine Learning Repository

The dataset contains 1,000 customer records with 20 input attributes and one target variable.

## 🤖 Machine Learning Models

The following classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest

## 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 78.0% | 66.67% | 53.33% | 59.26% | 80.40% |
| Decision Tree | 61.5% | 39.51% | 53.33% | 45.39% | 65.07% |
| Random Forest | 77.0% | 70.59% | 40.00% | 51.06% | 80.37% |

## 🖥️ Application Features

- Customer financial information input
- Credit risk prediction
- Good/Bad credit classification
- Bad credit probability
- Risk probability visualization
- Prediction explanation
- Feature contribution analysis
- Interactive Streamlit interface

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## 📁 Project Structure

```text
CodeAlpha_CreditScoringModel/
│
├── data/
│   └── german.data
│
├── notebooks/
│   └── 01_Data_Exploration.ipynb
│
├── models/
│   ├── credit_scoring_model.pkl
│   └── preprocessor.pkl
│
├── screenshots/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore