"""
Bike Rental Demand Prediction - Flask Web Application
======================================================
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Global variables to store loaded models
model = None
scaler = None
feature_names = None

# Load models on startup
def load_models():
    """Load the trained model and preprocessing objects"""
    global model, scaler, feature_names
    
    try:
        model_path = 'models/adaboost_bike_rental_model.pkl'
        scaler_path = 'models/minmax_scaler.pkl'
        features_path = 'models/feature_names.pkl'
        
        # Check if model files exist
        if not os.path.exists(model_path):
            print("⚠️  Model files not found. Please train and save models first.")
            return False
        
        # Load model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("✓ Model loaded successfully")
        
        # Load scaler
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        print("✓ Scaler loaded successfully")
        
        # Load feature names
        with open(features_path, 'rb') as file:
            feature_names = pickle.load(file)
        print("✓ Feature names loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return False


# Routes
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.get_json()
        
        # Extract features in correct order
        input_data = {
            'registered': int(data.get('registered', 0)),
            'season': int(data.get('season', 1)),
            'mnth': int(data.get('mnth', 1)),
            'hr': int(data.get('hr', 0)),
            'holiday': int(data.get('holiday', 0)),
            'weekday': int(data.get('weekday', 0)),
            'workingday': int(data.get('workingday', 0)),
            'weathersit': int(data.get('weathersit', 1))
        }
        
        # Create DataFrame with correct feature order
        df = pd.DataFrame([input_data])
        df = df[feature_names]
        
        # Scale the input
        scaled_input = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        predicted_count = int(prediction[0])
        
        # Prepare response
        response = {
            'success': True,
            'prediction': predicted_count,
            'input_data': input_data
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions from CSV upload"""
    try:
        # This would handle CSV file upload for batch predictions
        # Implementation can be added based on requirements
        return jsonify({
            'success': True,
            'message': 'Batch prediction feature - Coming soon!'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')


if __name__ == '__main__':
    print("=" * 70)
    print("🚴 BIKE RENTAL PREDICTION WEB APP")
    print("=" * 70)
    
    # Load models
    if load_models():
        print("\n✅ Models loaded successfully!")
        print(f"Expected features: {feature_names}")
        print("\n🌐 Starting Flask server...")
        print("=" * 70)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models. Please ensure model files exist in 'models/' directory.")
        print("Run the training notebook first to generate model files.")
