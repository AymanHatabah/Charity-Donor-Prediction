# 🎯 Charity Donor Prediction System

An end-to-end Machine Learning and web application project designed to predict whether an individual is likely to make a charitable donation based on census and demographic data. 

---

## 🚀 Live Demo
Experience the live application deployed on Streamlit Community Cloud:
👉 [View Live App](https://charity-donor-prediction-bsdzu3pfjeiumdaaei9fl.streamlit.app)

---

## 📊 Project Overview & Features
* **Exploratory Data Analysis (EDA):** Conducted rigorous univariate and multivariate analyses to extract deep insights into donor behavior and demographic patterns.
* **Advanced Feature Engineering:** Engineered custom transformers and leveraged `ColumnTransformer` for robust data preprocessing.
* **Handling Class Imbalance:** Integrated **SMOTE** (Synthetic Minority Over-sampling Technique) via `imbalanced-learn` to effectively tackle class skewness and enhance minority class recall.
* **Model Pipeline & Business Threshold:** Trained optimized classification pipelines with custom decision thresholds aligned with business objectives.
* **Data Validation:** Implemented strict schema validation using Pydantic.
* **Interactive Web App:** Built a sleek, user-friendly frontend using **Streamlit** for real-time predictions.

---

## 🛠️ Project Structure
```text
Charity-Donor-Prediction/
│
└── census_project/
    ├── app.py                          # Streamlit application UI
    ├── ml_utils.py                     # Utility & processing functions
    ├── custom_trasformer.py            # Custom preprocessing transformers
    ├── robust_donors_model_package.pkl # Serialized model pipeline & threshold
    ├── census.csv                      # Raw dataset
    ├── test_census.csv                 # Test dataset
    └── requirements.txt                # Project dependencies
