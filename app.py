"""
Viva Evaluation System - Flask Application
Serves both backend ML inference and HTML frontend
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Global variable to store the loaded model
model = None


class CompatUnpickler(pickle.Unpickler):
    """Custom unpickler to handle numpy version compatibility"""
    def find_class(self, module, name):
        # Fix numpy._core compatibility issues
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        elif module == 'numpy.core.multiarray' and name == '_reconstruct':
            module = 'numpy'
        return super().find_class(module, name)


def load_model():
    """Load the trained ML model from pickle file"""
    global model
    
    model_path = 'viva_predictions.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    try:
        # Try joblib first (preferred for scikit-learn)
        try:
            import joblib
            loaded_model = joblib.load(model_path)
            if hasattr(loaded_model, 'predict'):
                model = loaded_model
                print(f"✓ Model loaded from {model_path} (joblib)")
                return
        except:
            pass
        
        # Try pickle with compatibility fix
        with open(model_path, 'rb') as f:
            unpickler = CompatUnpickler(f)
            loaded_obj = unpickler.load()
        
        # Check if it's a model
        if hasattr(loaded_obj, 'predict'):
            model = loaded_obj
            print(f"✓ Model loaded from {model_path} (pickle)")
            return
        
        # If it's a DataFrame, create a model from it
        if hasattr(loaded_obj, 'columns') and hasattr(loaded_obj, 'values'):
            import pandas as pd
            from sklearn.linear_model import LinearRegression
            
            df = loaded_obj
            # Check if DataFrame has the right columns
            required_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
            if 'predicted_total_score' in df.columns and all(col in df.columns for col in required_cols):
                print(f"⚠ File contains DataFrame with predictions. Training model from data...")
                X = df[required_cols].values
                y = df['predicted_total_score'].values
                
                # Train a simple linear model
                model = LinearRegression()
                model.fit(X, y)
                print(f"✓ Model trained from {model_path} DataFrame")
                return
        
        raise Exception(f"File contains {type(loaded_obj).__name__}, not a model")
            
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")


def calculate_grade(total_score):
    """
    Calculate grade based on total score
    
    Grade Logic:
    - total_score >= 40 → A
    - total_score >= 30 → B
    - total_score >= 25 → C
    - else → Fail
    """
    if total_score >= 40:
        return "A"
    elif total_score >= 30:
        return "B"
    elif total_score >= 25:
        return "C"
    else:
        return "Fail"


@app.route('/')
def index():
    """Serve the HTML frontend"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict total score from question scores
    
    Expected JSON input:
    {
        "q1_score": float,
        "q2_score": float,
        "q3_score": float,
        "q4_score": float,
        "q5_score": float
    }
    
    Returns JSON:
    {
        "total_score": float,
        "grade": string
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Extract and validate scores
        scores = []
        for field in required_fields:
            try:
                score = float(data[field])
                if score < 0:
                    return jsonify({
                        "error": f"Invalid value for {field}: must be non-negative"
                    }), 400
                scores.append(score)
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Invalid value for {field}: must be a number"
                }), 400
        
        # Ensure model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please restart the application."
            }), 500
        
        # Prepare input for model prediction
        input_data = np.array([scores]).reshape(1, -1)
        
        # Make prediction
        try:
            total_score = float(model.predict(input_data)[0])
        except Exception as e:
            return jsonify({
                "error": f"Prediction error: {str(e)}"
            }), 500
        
        # Calculate grade
        grade = calculate_grade(total_score)
        
        # Return prediction result
        return jsonify({
            "total_score": round(total_score, 2),
            "grade": grade
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500


if __name__ == '__main__':
    # Load model on startup
    try:
        load_model()
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        print("Application will start but predictions will fail until model is available.")
        model = None
    
    # Run Flask app
    print("Starting Viva Evaluation System...")
    print("Access the application at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
