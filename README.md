# Viva Evaluation System

A Flask-based web application for predicting total viva scores and grades from individual question scores using machine learning.

## Features

- **ML-Powered Predictions**: Uses Linear Regression to predict total scores from 5 question scores
- **Web Interface**: Clean, responsive HTML frontend with real-time predictions
- **RESTful API**: JSON API endpoint for programmatic access
- **Automatic Grade Assignment**: Converts scores to grades (A, B, C, Fail)

## Requirements

- Python 3.8+
- Flask
- NumPy
- scikit-learn
- pandas
- joblib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Rubic-based-scoring-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the model file `viva_predictions.pkl` in the project root.

## Usage

### Start the Application

```bash
python app.py
```

The application will start on `http://127.0.0.1:5000`

### Using the Web Interface

1. Open your browser and navigate to `http://127.0.0.1:5000`
2. Enter scores for 5 questions (Q1-Q5)
3. Click "Predict Score"
4. View the predicted total score and grade

### Using the API

Send a POST request to `/predict`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "q1_score": 8,
    "q2_score": 7,
    "q3_score": 9,
    "q4_score": 8,
    "q5_score": 7
  }'
```

**Response:**
```json
{
  "total_score": 39.0,
  "grade": "B"
}
```

## Grade System

- **A**: Total score ≥ 40
- **B**: Total score ≥ 30
- **C**: Total score ≥ 25
- **Fail**: Total score < 25

## Project Structure

```
.
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── viva_predictions.pkl   # Model/predictions data file
├── templates/
│   └── index.html        # Frontend HTML
└── static/
    └── style.css         # Stylesheet
```

## API Endpoints

### `GET /`
Serves the HTML frontend.

### `POST /predict`
Predicts total score and grade from question scores.

**Request Body:**
```json
{
  "q1_score": float,
  "q2_score": float,
  "q3_score": float,
  "q4_score": float,
  "q5_score": float
}
```

**Response:**
```json
{
  "total_score": float,
  "grade": string
}
```

## Model Loading

The application automatically handles model loading:
- If `viva_predictions.pkl` contains a trained model, it loads directly
- If it contains a DataFrame with predictions, it trains a LinearRegression model from the data

## License

This project is for educational purposes.
