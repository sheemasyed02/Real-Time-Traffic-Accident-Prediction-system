import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.title('Traffic Accident Severity Prediction System')
st.caption('Predict accident severity based on road conditions and environmental factors')
st.markdown('---')

model_path = 'models/accident_severity_model.pkl'
if not os.path.exists(model_path):
    st.error('Models not found. Please train the models first.')
    st.stop()

model = joblib.load('models/accident_severity_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
encoders = joblib.load('models/label_encoders.pkl')
features = joblib.load('models/feature_names.pkl')

st.write('Enter accident parameters to predict severity')

st.subheader('Time and Location Details')
col1, col2 = st.columns(2)

with col1:
    hour = st.slider('Hour', 0, 23, 12)
    day = st.selectbox('Day', [1,2,3,4,5,6,7], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x-1])

with col2:
    cities = ['Mumbai','Delhi','Bangalore','Hyderabad','Chennai','Kolkata','Pune','Ahmedabad',
              'Jaipur','Surat','Lucknow','Kanpur','Nagpur','Indore','Thane','Bhopal',
              'Visakhapatnam','Pimpri-Chinchwad','Patna','Vadodara','Ghaziabad','Ludhiana',
              'Agra','Nashik','Faridabad','Meerut','Rajkot','Varanasi','Srinagar','Aurangabad',
              'Dhanbad','Amritsar','Navi Mumbai','Allahabad','Ranchi','Howrah','Coimbatore',
              'Jabalpur','Gwalior','Vijayawada','Jodhpur','Madurai','Raipur','Kota']
    city = st.selectbox('City', cities)
    
api_key = os.getenv('OPENWEATHER_API_KEY')
weather_info = None

if api_key and city:
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_info = data['weather'][0]['main']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        st.success(f"Current weather in {city}: {weather_info}, {temp:.1f}°C, Humidity {humidity}%")

st.subheader('Accident Details')

col3, col4 = st.columns(2)

with col3:
    units = st.number_input('Vehicles involved', 1, 10, 1)
    crash_type = st.selectbox('Crash type', ['ANGLE','REAR END','SIDESWIPE SAME DIRECTION','SIDESWIPE OPPOSITE DIRECTION',
                                               'TURNING','HEAD ON','FIXED OBJECT','PEDESTRIAN','PARKED MOTOR VEHICLE','OTHER OBJECT'])
    control = st.selectbox('Traffic control', ['NO CONTROLS','TRAFFIC SIGNAL','STOP SIGN/FLASHER','YIELD',
                                                'OTHER REG. SIGN','OTHER WARNING SIGN','POLICE/FLAGMAN','OTHER'])
    intersection = st.checkbox('Near intersection/junction')

with col4:
    weather = st.selectbox('Weather condition', ['CLEAR','RAIN','CLOUDY/OVERCAST','FOG/SMOKE/HAZE',
                                                   'SNOW','SLEET/HAIL','FREEZING RAIN/DRIZZLE','OTHER'])
    light = st.selectbox('Lighting', ['DAYLIGHT','DARKNESS','DARKNESS, LIGHTED ROAD','DAWN','DUSK'])
    surface = st.selectbox('Road surface', ['DRY','WET','SNOW OR SLUSH','ICE','SAND, MUD, DIRT','OTHER'])

st.markdown('---')

if st.button('Predict Severity', type='primary', use_container_width=True):
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
    
    base_severity = 0
    
    if crash_type in ['HEAD ON']:
        base_severity += 2.5
    elif crash_type in ['PEDESTRIAN']:
        base_severity += 2.0
    elif crash_type in ['FIXED OBJECT', 'PARKED MOTOR VEHICLE']:
        base_severity += 1.5
    elif crash_type in ['ANGLE']:
        base_severity += 1.2
    elif crash_type in ['REAR END']:
        base_severity += 0.8
    elif crash_type in ['SIDESWIPE SAME DIRECTION', 'SIDESWIPE OPPOSITE DIRECTION']:
        base_severity += 0.6
    else:
        base_severity += 0.3
    
    if weather in ['SNOW', 'FREEZING RAIN/DRIZZLE']:
        base_severity += 0.8
    elif weather in ['SLEET/HAIL', 'FOG/SMOKE/HAZE']:
        base_severity += 0.6
    elif weather == 'RAIN':
        base_severity += 0.3
    elif weather == 'CLOUDY/OVERCAST':
        base_severity += 0.1
    
    if surface == 'ICE':
        base_severity += 0.8
    elif surface == 'SNOW OR SLUSH':
        base_severity += 0.6
    elif surface == 'WET':
        base_severity += 0.3
    elif surface in ['SAND, MUD, DIRT', 'OTHER']:
        base_severity += 0.2
    
    if light in ['DARKNESS']:
        base_severity += 0.5
    elif light == 'DARKNESS, LIGHTED ROAD':
        base_severity += 0.3
    elif light in ['DAWN', 'DUSK']:
        base_severity += 0.2
    
    if units >= 4:
        base_severity += 0.8
    elif units == 3:
        base_severity += 0.5
    elif units == 2:
        base_severity += 0.2
    
    if control == 'NO CONTROLS':
        base_severity += 0.3
    
    base_severity = min(base_severity, 4.5)
    
    injuries_total = 0
    if base_severity >= 3.5:
        injuries_total = 3
    elif base_severity >= 2.0:
        injuries_total = 2
    elif base_severity >= 1.0:
        injuries_total = 1
    
    injuries_fatal = 1 if base_severity >= 3.8 else 0
    injuries_incapacitating = 1 if base_severity >= 2.5 else 0
    injuries_non_incapacitating = 1 if base_severity >= 1.2 else 0
    injuries_reported_not_evident = 1 if base_severity >= 0.6 else 0
    injuries_no_indication = 1 if base_severity < 0.6 else 0
    
    input_data = {
        'num_units': units,
        'injuries_total': injuries_total,
        'injuries_fatal': injuries_fatal,
        'injuries_incapacitating': injuries_incapacitating,
        'injuries_non_incapacitating': injuries_non_incapacitating,
        'injuries_reported_not_evident': injuries_reported_not_evident,
        'injuries_no_indication': injuries_no_indication,
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
        'crash_type': 'INJURY AND / OR TOW DUE TO CRASH' if base_severity >= 0.5 else 'NO INJURY / DRIVE AWAY',
        'damage': 'OVER $1,500' if base_severity >= 2.0 else '$501 - $1,500' if base_severity >= 0.8 else '$500 OR LESS',
        'prim_contributory_cause': 'EXCEEDING SAFE SPEED FOR CONDITIONS' if base_severity >= 2.5 else 'UNABLE TO DETERMINE',
        'intersection_related_i': 1 if intersection else 0
    }
    
    df = pd.DataFrame([input_data])
    
    for col in encoders:
        if col in df.columns:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except Exception as e:
                df[col] = 0
    
    for f in features:
        if f not in df.columns:
            df[f] = 0
    
    df = df[features]
    
    numeric_features = ['num_units','injuries_total','injuries_fatal','injuries_incapacitating',
                       'injuries_non_incapacitating','injuries_reported_not_evident','injuries_no_indication',
                       'Hour','Day_of_Week','Month','Year','Is_Weekend','Is_Rush_Hour','Season']
    
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    prediction = model.predict(df)[0]
    probs = model.predict_proba(df)[0]
    
    severity_labels = ['No Injury','Minor','Moderate','Serious','Fatal']
    severity_colors = ['Level 0','Level 1','Level 2','Level 3','Level 4']
    
    st.markdown('---')
    st.markdown(f"### Predicted Severity: {severity_labels[prediction]}")
    st.write(f"Severity Level: {prediction}")
    st.progress(float(probs[prediction]))
    st.write(f'Confidence: {probs[prediction]*100:.1f}%')
    
    st.write('')
    st.write('Probability for each severity level:')
    for i, label in enumerate(severity_labels):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.write(f'{label} ({severity_colors[i]})')
        with col_b:
            st.write(f'{probs[i]*100:.1f}%')
