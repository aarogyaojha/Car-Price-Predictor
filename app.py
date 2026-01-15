import streamlit as st
import pandas as pd
import pickle
import os

# Set absolute paths to ensure it runs from anywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_car.csv')

st.set_page_config(page_title="Car Price Predictor", layout="centered")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found! Please train the model using model/train.py first.")
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def main():
    st.title("🚗 Car Price Predictor")
    st.markdown("### Predict the fair market value of used cars using our LinearRegressionModel")
    
    model = load_model()
    
    if model and os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
        
        st.divider()
        st.subheader("Enter Car Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            companies = sorted(data['company'].unique())
            company = st.selectbox("Company", companies)
            
            # Filter models by selected company
            models = sorted(data[data['company'] == company]['name'].unique())
            name = st.selectbox("Car Model", models)
            
        with col2:
            years = sorted(data['year'].unique(), reverse=True)
            year = st.selectbox("Year of Purchase", years)
            
            fuel_types = data['fuel_type'].unique()
            fuel_type = st.selectbox("Fuel Type", fuel_types)
            
        kms_driven = st.number_input("Kilometers Driven", min_value=0, value=10000, step=1000)
        
        st.divider()
        
        if st.button("Predict Price", type="primary"):
            # Prepare input data matching the model's training columns
            input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]], 
                                      columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
            
            try:
                prediction = model.predict(input_data)
                
                # Display result
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                    <h2>Predicted Price</h2>
                    <h1 style="color: #00CC00;">₹ {prediction[0]:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                
    else:
        if not os.path.exists(DATA_PATH):
            st.error("Data file not found! Please ensure data/cleaned_car.csv exists.")

if __name__ == "__main__":
    main()
