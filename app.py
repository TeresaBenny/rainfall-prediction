from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__,static_folder='static')

# Load the trained model and scaler
model = joblib.load("best_rainfall_prediction_model_XGBoost.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature order
feature_names = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                 'Pressure3pm', 'Cloud9am', 'Cloud3pm']

@app.route('/')
def landing():
    """Landing page that directs to prediction page."""
    return render_template('landing.html')

@app.route('/predict_form')
def home():
    """Renders the main prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and makes predictions."""
    try:
        # Get form data and convert to float
        data = [float(request.form[feature]) for feature in feature_names]
        
        # Convert to NumPy array and scale
        new_data_scaled = scaler.transform([data])
        
        # Make prediction
        prediction = model.predict(new_data_scaled)[0]
        
        # Redirect to different pages based on prediction
        if prediction == 1:
            # If prediction is 1, go to rain.html
            return render_template("rain.html")
        else:
            # If prediction is 0, go to no_rain.html
            return render_template("no_rain.html")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

