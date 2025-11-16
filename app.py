import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

model = joblib.load('models/accident_severity_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
encoders = joblib.load('models/label_encoders.pkl')
features = joblib.load('models/feature_names.pkl')

st.title('Traffic Accident Severity Prediction')
st.write('Predict severity based on road and weather conditions')

st.subheader('Time and Location')
col1, col2 = st.columns(2)

with col1:
    hour = st.slider('Hour', 0, 23, 12)
    day = st.selectbox('Day', [1,2,3,4,5,6,7], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x-1])

with col2:
    cities = ['Istanbul','Ankara','Izmir','Bursa','Antalya','Adana','Gaziantep','Konya',
              'Kayseri','Mersin','Eskisehir','Diyarbakir','Samsun','Denizli','Trabzon',
              'Malatya','Erzurum','Van']
    city = st.selectbox('City', cities)
    
api_key = os.getenv('OPENWEATHER_API_KEY')
weather_info = None

if api_key and city:
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_info = data['weather'][0]['main']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        st.success(f"Weather in {city}: {weather_info}, {temp:.1f}°C, Humidity {humidity}%")

st.subheader('Accident Details')

col3, col4 = st.columns(2)

with col3:
    units = st.number_input('Vehicles involved', 1, 10, 2)
    crash_type = st.selectbox('Crash type', ['REAR END','SIDESWIPE','ANGLE','HEAD ON','PARKED VEHICLE'])
    control = st.selectbox('Traffic control', ['NO CONTROLS','TRAFFIC SIGNAL','STOP SIGN','YIELD SIGN'])

with col4:
    weather = st.selectbox('Weather condition', ['CLEAR','RAIN','SNOW','CLOUDY','FOG'])
    light = st.selectbox('Lighting', ['DAYLIGHT','DARKNESS','DARKNESS LIGHTED','DUSK','DAWN'])
    surface = st.selectbox('Road surface', ['DRY','WET','SNOW OR SLUSH','ICE'])

if st.button('Predict Severity'):
    month = datetime.now().month
    year = datetime.now().year
    
    is_weekend = 1 if day >= 6 else 0
    is_rush = 1 if hour in [7,8,9,16,17,18] else 0
    
    if month in [12,1,2]:
        season = 1
    elif month in [3,4,5]:
        season = 2
    elif month in [6,7,8]:
        season = 3
    else:
        season = 4
    
    input_data = {
        'num_units': units,
        'injuries_total': 0,
        'injuries_fatal': 0,
        'injuries_incapacitating': 0,
        'injuries_non_incapacitating': 0,
        'injuries_reported_not_evident': 0,
        'injuries_no_indication': 0,
        'Hour': hour,
        'Day_of_Week': day,
        'Month': month,
        'Year': year,
        'Is_Weekend': is_weekend,
        'Is_Rush_Hour': is_rush,
        'Season': season,
        'weather_condition': weather,
        'lighting_condition': light,
        'traffic_control_device': control,
        'first_crash_type': crash_type,
        'trafficway_type': 'NOT DIVIDED',
        'alignment': 'STRAIGHT AND LEVEL',
        'roadway_surface_cond': surface,
        'road_defect': 'NO DEFECTS',
        'crash_type': 'INJURY AND / OR TOW DUE TO CRASH',
        'damage': 'OVER $1,500',
        'prim_contributory_cause': 'UNABLE TO DETERMINE',
        'intersection_related_i': 0
    }
    
    df = pd.DataFrame([input_data])
    
    for col in encoders:
        if col in df.columns:
            try:
                df[col] = encoders[col].transform(df[col])
            except:
                df[col] = 0
    
    for f in features:
        if f not in df.columns:
            df[f] = 0
    
    df = df[features]
    
    numeric_features = ['num_units','injuries_total','injuries_fatal','injuries_incapacitating',
                       'injuries_non_incapacitating','injuries_reported_not_evident','injuries_no_indication',
                       'Hour','Day_of_Week','Month','Year','Is_Weekend','Is_Rush_Hour']
    
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    prediction = model.predict(df)[0]
    probs = model.predict_proba(df)[0]
    
    severity_labels = ['No Injury','Non-Incapacitating','Incapacitating','Fatal']
    colors = ['green','orange','red','darkred']
    
    st.markdown('---')
    st.subheader(f'Predicted Severity: {severity_labels[prediction]}')
    st.write(f'Confidence: **{probs[prediction]*100:.1f}%**')
    
    st.write('**Probability for each severity level:**')
    for i, label in enumerate(severity_labels):
        st.write(f'{label}: {probs[i]*100:.1f}%')
