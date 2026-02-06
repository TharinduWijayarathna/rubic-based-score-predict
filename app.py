"""
Viva Evaluation System - Flask Application
Serves both backend ML inference and HTML frontend
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Global variable to store the loaded model
model = None


def load_model():
    """Load the trained ML model from pickle file"""
    global model
    # Try primary filename first, then fallback to alternative
    model_paths = ['viva_scoring_model.pkl', 'viva_predictions.pkl']
    model_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError(
            f"Model file not found. Tried: {', '.join(model_paths)}. "
            "Please ensure the model file exists in the project root."
        )
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded successfully from {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


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
            return jsonify({
                "error": "No JSON data provided"
            }), 400
        
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
            score = data[field]
            
            # Check if score is numeric
            try:
                score = float(score)
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Invalid value for {field}: must be a number"
                }), 400
            
            # Check if score is non-negative (assuming scores can't be negative)
            if score < 0:
                return jsonify({
                    "error": f"Invalid value for {field}: must be non-negative"
                }), 400
            
            scores.append(score)
        
        # Ensure model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please restart the application."
            }), 500
        
        # Prepare input for model prediction
        # Assuming model expects a 2D array (single sample)
        input_data = [scores]
        
        # Make prediction
        try:
            total_score = model.predict(input_data)[0]
            # Ensure total_score is a float
            total_score = float(total_score)
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
        print(f"✗ Error: {str(e)}")
        print("Application will start but predictions will fail until model is available.")
    
    # Run Flask app
    print("Starting Viva Evaluation System...")
    print("Access the application at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
