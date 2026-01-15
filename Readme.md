# Car Price Predictor

## Project Overview
This project establishes a machine learning pipeline to predict the selling price of used cars. It encompasses data cleaning, feature engineering, model training using a LinearRegressionModel, and a web-based demonstration application.

## Problem Statement
The used car market suffers from pricing opacity. This project aims to bring transparency by estimating the fair market value of a car based on its attributes, utilizing a regression approach to model the relationship between vehicle specifications and market price.

## Dataset Description
The dataset comprises listings of used cars with the following attributes:
- **name**: The make and model of the car.
- **company**: The manufacturer.
- **year**: The year of manufacture/purchase.
- **Price**: The listed selling price (Target Variable).
- **kms_driven**: The distance the car has been driven in kilometers.
- **fuel_type**: The type of fuel used (Petrol, Diesel, LPG).

## Feature Explanation
- **Categorical Features**: `name`, `company`, `fuel_type` are nominal variables encoded using OneHotEncoding.
- **Numerical Features**: `year`, `kms_driven` are treated as continuous variables.
- **Target**: `Price` is the continuous target variable for regression.

## Model Selection: LinearRegressionModel
The LinearRegressionModel was selected for this project due to:
1. **Interpretability**: The model coefficients provide direct insight into the price impact of each feature.
2. **Generalization**: It performs robustly with OneHotEncoded sparse matrices derived from categorical features.
3. **Efficiency**: Training and inference are computationally inexpensive for this dataset size.

## Training & Evaluation Metrics
- **Algorithm**: Linear Regression (Ordinary Least Squares).
- **Preprocessing**: OneHotEncoding for categorical variables; standard cleaning for numerical inputs.
- **Metric**: R2 Score (Coefficient of Determination) is used to evaluate the model performance on the test set.

## How to Run the Notebook
1. Navigate to the `notebooks` directory.
2. Ensure `../data/quikr_car.csv` exists.
3. Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook car_price_predictor.ipynb
   ```
4. Execute cells sequentially.

## How to Train the LinearRegressionModel
To retrain the model from the command line:
1. Navigate to the project root `car-price-predictor`.
2. Run the training script:
   ```bash
   python model/train.py
   ```
3. This will generate `model/model.pkl` and `data/cleaned_car.csv`.

## How to Run the Streamlit App
To launch the interactive prediction application:
1. Navigate to the project root `car-price-predictor`.
2. Run the Streamlit server:
   ```bash
   streamlit run app.py
   ```
3. Access the application in your web browser at the provided local URL (typically http://localhost:8501).

## Limitations
- The model assumes a linear relationship between features and price, which may simplify complex market dynamics.
- The dataset quality required significant cleaning; potential hidden biases in the missing data remain.
- The model is limited to the brands and models present in the training data.