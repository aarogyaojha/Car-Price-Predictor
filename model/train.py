import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'quikr_car.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
CLEANED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_car.csv')

def train_model():
    """
    Trains the LinearRegressionModel using the dataset.
    Mirroring the logic from car_price_predictor.ipynb exactly.
    """
    print("Loading data...")
    car = pd.read_csv(DATA_PATH)
    
    # Preprocessing (Exact replica of notebook)
    # 1. Backup
    backup = car.copy()
    
    # 2. Year cleaning
    car = car[car['year'].str.isnumeric()]
    car['year'] = car['year'].astype(int)
    
    # 3. Price cleaning
    car = car[car['Price'] != "Ask For Price"]
    car['Price'] = car['Price'].str.replace(',', '').astype(int)
    
    # 4. Kms Driven cleaning
    car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')
    car = car[car['kms_driven'].str.isnumeric()]
    car['kms_driven'] = car['kms_driven'].astype(int)
    
    # 5. Fuel Type cleaning
    car = car[~car['fuel_type'].isna()]
    
    # 6. Name cleaning (First 3 words)
    car['name'] = car['name'].str.split(' ').str.slice(0,3).str.join(' ')
    
    # 7. Reset index
    car = car.reset_index(drop=True)
    
    # 8. Outlier removal
    car = car[car['Price']<6e6].reset_index(drop=True)
    
    # Save cleaned data
    car.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {CLEANED_DATA_PATH}")
    
    # Model Training
    X = car.drop(columns='Price')
    y = car['Price']
    
    # Train Test Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=433)
    
    # OneHotEncoding
    ohe = OneHotEncoder()
    ohe.fit(X[['name','company','fuel_type']])
    
    # Column Transformer
    column_trans = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), ['name','company','fuel_type']),
        remainder='passthrough'
    )
    
    # Linear Regression Model
    lr = LinearRegression()
    
    # Pipeline
    pipe = make_pipeline(column_trans, lr)
    
    print("Training LinearRegressionModel...")
    pipe.fit(X_train, y_train)
    
    # Save model
    pickle.dump(pipe, open(MODEL_PATH, "wb"))
    print(f"LinearRegressionModel trained and saved to {MODEL_PATH}")
    
    # Basic Evaluation
    from sklearn.metrics import r2_score
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2}")

if __name__ == "__main__":
    train_model()
