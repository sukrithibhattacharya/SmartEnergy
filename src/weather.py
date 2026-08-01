import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fetch_weather_data(lat=28.6139, lon=77.2090, start_date=None, end_date=None):
    """Fetch weather data from Open-Meteo"""
    if start_date is None:
        start_date = '2024-01-01'
    if end_date is None:
        end_date = '2024-12-31'
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "weathercode", "wind_speed_10m_max"],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'daily' not in data:
        return None
    
    daily = data['daily']
    df = pd.DataFrame({
        'date': pd.to_datetime(daily['time']),
        'temp_max': daily['temperature_2m_max'],
        'temp_min': daily['temperature_2m_min'],
        'weather_code': daily['weathercode'],
        'wind_speed': daily['wind_speed_10m_max']
    })
    df.set_index('date', inplace=True)
    return df

def encode_weather(weather_df):
    """Encode weather features for model input"""
    def weather_type(code):
        if code in [0, 1]:
            return 0  # clear
        elif code in [2, 3, 45, 48, 51, 53, 55, 61, 63, 65, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99]:
            return 2  # rainy
        else:
            return 1  # cloudy
    
    weather_df['weather_type'] = weather_df['weather_code'].apply(weather_type)
    
    scaler = MinMaxScaler()
    num_scaled = scaler.fit_transform(weather_df[['temp_max', 'temp_min', 'wind_speed']].values)
    onehot = pd.get_dummies(weather_df['weather_type'], prefix='wt').values
    
    return np.hstack([num_scaled, onehot]), scaler