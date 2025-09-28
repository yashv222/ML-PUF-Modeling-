## 📌 Overview
This project was developed as part of **CS771 (Synergy Project), IIT Kanpur (Jan–Apr 2024)**.  
It explores **machine learning techniques for modeling Physical Unclonable Functions (PUFs)** and **classification of Android malware**.  

The work combines **mathematical derivations, feature engineering, and ML modeling** to achieve robust classification accuracy.

## ✨ Key Features
- **PUF Modeling**
  - Derived recurrence relations for Arbiter PUFs.
  - Engineered a **255-dimensional Boolean feature map**.
  - Proved XOR of two PUFs reducible to a single linear model.

- **Malware Classification**
  - Preprocessed Android malware dataset with **feature reduction (30 features via Random Forest)**.
  - Implemented ML models: **Random Forest (RF), Support Vector Machine (SVM), Decision Tree (DT), Logistic Regression (LR)**.
  - Achieved **100% accuracy with RF & SVM**.
  - ## 🛠 Tech Stack
- **Languages:** Python (NumPy, Pandas, Scikit-learn), Jupyter Notebooks
- **ML Models:** Random Forest, SVM, Logistic Regression, Decision Tree
- **Concepts:** Physical Unclonable Functions (PUFs), Arbiter PUFs, XOR PUF, Feature Map Expansion

📊 Results

PUF Modeling: Reduced XOR of 2 PUFs → Single Linear Model in 255-d Boolean space.

Malware Classification:

RF & SVM achieved 100% accuracy.

LR & DT reached high accuracy after feature reduction.
