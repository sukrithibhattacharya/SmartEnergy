import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ----------------- Weather Functions -----------------

def fetch_weather_data(lat=28.6139, lon=77.2090, start_date=None, end_date=None):
    """
    Fetch historical daily weather from Open-Meteo ERA5 archive.
    Returns DataFrame or None if fails.
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "weathercode", "wind_speed_10m_max"],
        "timezone": "auto"
    }
    try:
        print(f"Fetching weather from ERA5: {start_date} to {end_date}")
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        if 'daily' not in data:
            print("No 'daily' field in response.")
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
        print(f"Successfully got {len(df)} days of weather.")
        return df
    except Exception as e:
        print(f"Weather fetch failed: {e}")
        return None

def generate_synthetic_weather(start_date, end_date):
    """
    Fallback: Generate realistic synthetic weather for Delhi.
    """
    print("Using synthetic weather as fallback.")
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)
    month_temp = {
        1: (5, 20), 2: (8, 25), 3: (15, 30), 4: (22, 36), 5: (26, 40), 6: (28, 39),
        7: (26, 35), 8: (25, 34), 9: (23, 34), 10: (18, 33), 11: (12, 28), 12: (7, 22)
    }
    temps, codes, winds = [], [], []
    for d in date_range:
        tmin, tmax = month_temp[d.month]
        tmin += np.random.uniform(-3, 3)
        tmax += np.random.uniform(-3, 3)
        temps.append([tmax, tmin])
        code = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        codes.append(code)
        winds.append(np.random.uniform(5, 25))
    df = pd.DataFrame({
        'date': date_range,
        'temp_max': [t[0] for t in temps],
        'temp_min': [t[1] for t in temps],
        'weather_code': codes,
        'wind_speed': winds
    })
    df.set_index('date', inplace=True)
    return df

def encode_weather(df_weather):
    """Encode weather: normalize numeric, one-hot encode type."""
    def wtype(code):
        if code in [0, 1]:
            return 0      # clear
        elif code in [2, 3, 45, 48, 51, 53, 55, 61, 63, 65, 71, 73, 75, 77,
                      80, 81, 82, 85, 86, 95, 96, 99]:
            return 2      # rainy/snow
        else:
            return 1      # cloudy
    df_weather['weather_type'] = df_weather['weather_code'].apply(wtype)
    scaler_num = MinMaxScaler()
    num_scaled = scaler_num.fit_transform(df_weather[['temp_max', 'temp_min', 'wind_speed']].values)
    onehot = pd.get_dummies(df_weather['weather_type'], prefix='wt').values
    features = np.hstack([num_scaled, onehot])
    return features, scaler_num

# ----------------- Load Data -----------------

def load_and_resample_load(file_path, resample_rule='15T'):
    df = pd.read_csv(file_path)
    # Find timestamp column
    if 'timestamp' in df.columns:
        ts_col = 'timestamp'
    elif 'datetime' in df.columns:
        ts_col = 'datetime'
    else:
        ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df.set_index(ts_col, inplace=True)
    df.sort_index(inplace=True)
    # Find load column
    if 'load' in df.columns:
        load_col = 'load'
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            load_col = numeric_cols[0]
        else:
            raise ValueError("No numeric load column found.")
    df = df[[load_col]].astype(float)
    # Resample to 15-min intervals
    df_resampled = df.resample(resample_rule).interpolate(method='linear')
    df_resampled.dropna(inplace=True)
    return df_resampled[load_col]

# ----------------- Main Preparation -----------------

def prepare_data(load_file, lat=28.6139, lon=77.2090, 
                 history_days=7, points_per_day=96, test_ratio=0.1):
    """
    Complete data preparation pipeline:
    - Loads load data
    - Fetches weather data
    - Creates sliding windows
    - Aligns weather to targets
    - Normalizes everything
    Returns ready-to-use numpy arrays
    """
    # 1. Load load series
    load_series = load_and_resample_load(load_file)
    start_date = load_series.index.min().normalize()
    end_date = load_series.index.max().normalize()
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # 2. Fetch weather (ERA5) or fallback
    weather_raw = fetch_weather_data(lat, lon, start_str, end_str)
    if weather_raw is None:
        print("ERA5 fetch failed. Using synthetic weather.")
        weather_raw = generate_synthetic_weather(start_date, end_date)

    # 3. Build sliding windows
    window_size = history_days * points_per_day
    data = load_series.values
    target_dates = []
    X_load = []
    y = []
    
    for i in range(window_size, len(data) - points_per_day, points_per_day):
        seq = data[i - window_size : i]
        target_day = data[i : i + points_per_day]
        avg = np.mean(target_day)
        maxv = np.max(target_day)
        minv = np.min(target_day)
        X_load.append(seq)
        y.append([avg, maxv, minv])
        target_date = load_series.index[i].normalize()
        target_dates.append(target_date)
    
    X_load = np.array(X_load)
    y = np.array(y)

    # 4. Align weather to each target date
    weather_features = []
    for d in target_dates:
        if d in weather_raw.index:
            row = weather_raw.loc[d]
            weather_features.append([row['temp_max'], row['temp_min'], row['weather_code'], row['wind_speed']])
        else:
            idx = weather_raw.index.get_indexer([d], method='nearest')[0]
            if idx >= 0:
                row = weather_raw.iloc[idx]
                weather_features.append([row['temp_max'], row['temp_min'], row['weather_code'], row['wind_speed']])
            else:
                mean_vals = weather_raw.mean()
                weather_features.append([mean_vals['temp_max'], mean_vals['temp_min'], mean_vals['weather_code'], mean_vals['wind_speed']])
    
    weather_df = pd.DataFrame(weather_features, columns=['temp_max', 'temp_min', 'weather_code', 'wind_speed'])
    weather_encoded, scaler_weather_num = encode_weather(weather_df)

    # 5. Train/test split
    split_idx = int(len(X_load) * (1 - test_ratio))
    X_load_train, X_load_test = X_load[:split_idx], X_load[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    weather_train, weather_test = weather_encoded[:split_idx], weather_encoded[split_idx:]

    # 6. Scale features
    scaler_X = MinMaxScaler()
    X_load_train_flat = X_load_train.reshape(-1, 1)
    X_load_test_flat = X_load_test.reshape(-1, 1)
    X_load_train_scaled_flat = scaler_X.fit_transform(X_load_train_flat)
    X_load_test_scaled_flat = scaler_X.transform(X_load_test_flat)
    X_load_train_scaled = X_load_train_scaled_flat.reshape(X_load_train.shape)
    X_load_test_scaled = X_load_test_scaled_flat.reshape(X_load_test.shape)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Reshape to (samples, history_days, points_per_day) for BiLSTM
    X_load_train_scaled = X_load_train_scaled.reshape(-1, history_days, points_per_day)
    X_load_test_scaled = X_load_test_scaled.reshape(-1, history_days, points_per_day)

    return (X_load_train_scaled, X_load_test_scaled,
            weather_train, weather_test,
            y_train_scaled, y_test_scaled,
            scaler_y, scaler_X, scaler_weather_num)