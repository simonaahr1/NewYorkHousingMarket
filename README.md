# NewYorkPropertyPrices
# Overview
This project is build with data from: https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market/data
This project predicts house prices in New York City using three different machine learning models: Linear Regression, Random Forest, and Neural Network.
It includes full data preprocessing, feature engineering, model training, evaluation, and prediction on new data.

# Features
Advanced Feature Engineering:
ZIP code extraction, borough/region detection, distances to key landmarks (city center, Central Park), and categorical encoding.

Multiple ML Models:

-Linear Regression

-Random Forest

-Neural Network (Keras)

Robust Preprocessing:

-Outlier removal

-Log transformation of target variable

-Target encoding for categorical features

-Scaling for neural network

Model Evaluation:
Metrics: MSE, MAE, and R² score.

# How to Run
1. Install requirements

  pip install -r requirements.txt

2. Prepare your data
Put your dataset in the data/ folder, named NY-House-Dataset.csv.

3. You can use the notebooks/EDA to check the analysis and why did we make some features

4. Train all models
python main.py

5. Make a prediction on a new property
Edit the example input in prediction.py or import the functions in your own script

# Preprocessing & Feature Engineering
Log-transform of price to reduce skew and improve model performance.

Borough and ZIP extraction from address/state fields.

Distance calculations from property to city center and Central Park.

Target encoding for categorical features.

Outlier removal on log price, beds, baths, and sqft.

# Model Evaluation
Each model’s performance is reported with MSE, MAE, and R² score (see console output after running main.py).

Detailed error analysis and prediction vs. actual exported as CSV for further analysis.

