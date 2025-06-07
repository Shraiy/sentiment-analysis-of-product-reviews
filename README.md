# Sentiment Classification of Product Reviews

This project implements the NLP techniques and fine-tuned LLM models to extract meaningful sentiment from e-commerce reviews in order to understand customer perception and improve decision-making.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![NLP](https://img.shields.io/badge/NLP-SentimentAnalysis-yellow)
![Transformer Models](https://img.shields.io/badge/Models-DistilBERT%20|%20RoBERTa-orange)

## 📌 Project Overview

This project implements a robust sentiment classification framework for product reviews, using both **traditional machine learning models** and **pre-trained transformer models**. The goal is to classify customer reviews into **positive**, **neutral**, or **negative** sentiment classes.

This work was developed as part of a Data Science Internship at **Saanvi Innovative Solutions** during Jan–June 2025.

---

## 🔍 Problem Statement

Online product reviews play a critical role in shaping consumer opinions and business decisions. However, extracting sentiment from unstructured text poses significant challenges. This project compares traditional ML methods (Naive Bayes, Random Forest) with state-of-the-art transformer models (DistilBERT, RoBERTa) to determine the most effective approach for multiclass sentiment classification.

---

## 🧰 Technologies Used

- **Python** (3.10)
- **Jupyter Notebook**
- **Pandas, NumPy**
- **Scikit-learn**
- **NLTK**
- **Matplotlib, Seaborn**
- **Transformers (HuggingFace)**
- **DistilBERT, RoBERTa**
- **SMOTE (imbalanced-learn)**

---

## 🗂️ Dataset

- **Source**: [Flipkart Product Reviews - Kaggle](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset)
- **Features Used**: `Review`, `Summary`, `Rating`
- **Target**: Sentiment (`Positive`, `Neutral`, `Negative`)

---

## 🔄 Pipeline & Methodology

### 🔹 Data Preprocessing

- Lowercasing, stopword removal, lemmatization
- Feature engineering combining review + summary
- Label encoding for sentiment classes
- Data balancing using **SMOTE** and **Random Oversampling**

### 🔹 Model Development

1. **Traditional ML Models**
   - TF-IDF Vectorization
   - Multinomial Naive Bayes (MNB)
   - Random Forest Classifier

2. **Transformer Models**
   - Tokenization using HuggingFace Tokenizers
   - Fine-tuning on:
     - `DistilBERT`
     - `RoBERTa`

### 🔹 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Visualizations (word clouds, performance comparison)

---

## 📊 Results

| Model           | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Naive Bayes     | 88.48%   | 69.10%    | 75.83% | 71.33%   |
| Random Forest   | 92.20%   | 76.55%    | 74.80% | 75.58%   |
| DistilBERT      | 98.39%   | 98.40%    | 98.39% | 98.39%   |
| RoBERTa         | 97.81%   | 97.82%    | 97.81% | 97.81%   |

> 🚀 DistilBERT outperformed all models, showcasing the strength of transformer-based architectures.

---

## 📌 Future Enhancements

- Incorporate LSTM/GRU and hybrid models
- Deploy sentiment classifier via web or mobile UI
- Apply Explainable AI techniques (e.g., SHAP, LIME)
- Extend support for multilingual reviews
- Handle sarcasm and contextual ambiguities

---
