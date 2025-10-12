# CLARION: **C**hurn **L**earning with **A**I-driven **R**easoning and **I**nterpretati**ON**

This project, developed for Megathon '25, addresses the significant revenue loss and lack of clear insight associated with customer churn. Our solution, CLARION, provides a robust Churn Prediction model to accurately identify at-risk customers, coupled with Interpretability using SHAP and DeepSHAP to pinpoint the exact drivers of churn. The goal is to maximize the ROI of retention efforts by enabling targeted, data-driven strategies.

**DATASET:** [Auto Insurance Churn Analysis (Kaggle)](https://www.kaggle.com/datasets/merishnasuwal/auto-insurance-churn-analysis-dataset)

## Run Demo

```
cd xgboost-app
pip install -r requirements.txt
streamlit run app.py
```

## Data Preprocessing

```
python data-preprocess.py
```

To handle the dataset's class imbalance and other issues, preprocessing involved:

- Data Cleaning
- Filling Missing Values
- Feature Engineering
- Data Imputation
- Normalisation
- SMOTE

## Baselines

```
python baselines/logistic-regression.py
python baselines/mlp.py
```

The Logistic Regression baseline was significantly improved by preprocessing and SMOTE, demonstrating a dramatic rise in Recall. The Multi-Layer Perceptron (MLP) showed a further jump in overall performance.

## XGBoost

Open and Run All:

```
xgboost/xgboost.ipynb
```

XGBoost was chosen as the high-performance benchmark, excelling at handling complex interactions and scaling. It achieved superior performance metrics: Accuracy 0.88, Precision 0.70, Recall 0.69, and F1-score 0.70. The model's predictions are explained using SHAP values, which identified county, days_tenure, and city as the top three features impacting churn.

## TabNet

```
python tabnet/train.py
```

TabNet is an Attention-based Deep Learning model specifically designed for tabular data, providing local, interpretable feature selection through attention masks. We utilized DeepSHAP for its post-hoc interpretability.
