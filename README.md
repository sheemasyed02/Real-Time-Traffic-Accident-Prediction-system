# Traffic Accident Severity Prediction System

Machine learning system to predict accident severity based on road conditions, weather, and traffic data.

## Dataset

**Source:** Turkey Traffic Accidents (Kaggle)
- Records: 200,000+ accidents
- Features: Weather, time, road conditions, crash type
- Target: Severity (No Injury, Non-Incapacitating, Incapacitating, Fatal)

## Installation

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Setup weather API (optional):
```
Copy .env.example to .env
Add your OpenWeather API key: OPENWEATHER_API_KEY=api_key
```

Get free key at: https://openweathermap.org/api

## Usage

### Train Models

Run the notebook to train models:
```
jupyter notebook TrafficAccident.ipynb
```

This will:
- Load and clean data
- Create time-based features (hour, day, season, rush hour)
- Encode categorical variables
- Train 3 models (Random Forest, XGBoost, Gradient Boosting)
- Save best model to `models/` folder

Training takes about 15 minutes.

### Run Web App

Start the Streamlit app:
```
streamlit run app.py
```

App opens at http://localhost:8501

## Features

**Time Features:**
- Hour, day of week, month
- Weekend
- Rush hour(7-9 AM, 4-6 PM)
- Season

**Road Conditions:**
- Weather condition
- Lighting
- Road surface
- Traffic control devices

**Accident Details:**
- Number of vehicles
- Crash type
- Intersection flag

## Models

Three models trained and compared:
- Random Forest
- XGBoost (gradient boosting)
- Gradient Boosting

Best model automatically selected based on test accuracy.

## Tech Stack

- Python 3.12
- pandas, numpy
- scikit-learn
- XGBoost
- SMOTE (for class balancing)
- Streamlit
- matplotlib, seaborn

## Results

All models achieved high accuracy on test data. The notebook includes visualizations for:
- Severity distribution
- Hourly accident patterns
- Model accuracy comparison
- Feature importance
- Confusion matrix
