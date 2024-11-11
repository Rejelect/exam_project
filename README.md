
---

# Project Name: Health Status Prediction

This project leverages machine learning models to predict health-related outcomes based on various physiological and lifestyle features. It incorporates advanced feature engineering and model tuning techniques to achieve high prediction accuracy.



### Project Overview

The goal of this project is to accurately predict a health status indicator using machine learning. The project applies feature engineering techniques to enhance predictive performance and includes models like Logistic Regression, Random Forest, and Stacking Classifier for optimized accuracy.

### Dataset

The dataset contains various health-related features, including:

- Physiological Metrics: hemoglobin, height(cm), weight(kg), Gtp, ALT, and others.
- Derived Metrics: BMI, hemoglobin-related calculations, age-adjusted categories.
- Target Variable: smoking (binary classification)

The data is preprocessed to handle missing values, encode categorical variables, and scale numerical features.

### Feature Engineering

Feature engineering plays a critical role in this project and includes:

- BMI Calculation: Body Mass Index computed as weight(kg) / (height(cm) / 100) ** 2.
- **Hemoglobin Adjusted Features**:
  - hemoglobin_height: Product of hemoglobin and height(cm).
  - hemoglobin / Gtp: Ratio of hemoglobin to Gtp.
  - hemoglobin_status: Indicator based on hemoglobin levels within age-specific ranges.
- Categorical Binning:
  - ALT_binned_quantile: Discretizes ALT levels into quantiles.
  - BMI_category: Categorizes BMI into standard ranges.

These engineered features are designed to capture meaningful relationships and improve model accuracy.

### Modeling

The model training and evaluation process utilizes several classifiers:

- Logistic Regression: Baseline model for linear classification.
- Random Forest: Tuned with hyperparameters to optimize depth, number of estimators, and split criteria.
- Stacking Classifier: Combines multiple classifiers (Bagging and Logistic Regression) for robust performance.

Additionally, Optuna is used for hyperparameter optimization, focusing on maximizing the ROC-AUC score. Calibration techniques are applied to improve the probability predictions, and both Brier Score and ROC-AUC are computed to assess model performance.

### Installation

To set up the project environment, install the required dependencies:
```
pip install -r requirements.txt
```
Dependencies include packages for data processing (pandas, numpy), visualization (seaborn, matplotlib), machine learning (scikit-learn), and optimization (optuna).

### Usage

1. Data Preprocessing:
   - Use feature engineering functions in the notebook or engineer.py to transform the dataset.
   - Create engineered features and split the data for model training and testing.

2. Model Training and Evaluation:
   - Train the model using either the provided notebook or load the pre-trained model (model.pkl).
   - Evaluate performance with metrics like ROC-AUC and Brier Score.

Example:
```python
from engineer import FeatureTransformer
import pandas as pd
from joblib import load

# Load and preprocess data
data = pd.read_csv('your_data.csv')
transformer = FeatureTransformer()
processed_data = transformer.transform(data)

# Load model and predict
model = load('model.pkl')
predictions = model.predict(processed_data)
```
### Results
The results and evaluation metrics are documented in the analyse.ipynb, analys.ipynb, and model.ipynb notebooks. Visualizations, feature importance plots, and calibration plots are also included to provide insights into model performance and prediction reliability.
