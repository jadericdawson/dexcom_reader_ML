import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import requests
import numpy as np
import pandas as pd
import pytz
from requests.exceptions import RequestException
from dotenv import load_dotenv
import os
import logging
from time import sleep
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import pickle

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("glucose_monitor.log"), logging.StreamHandler()]
)

# ---------------------------
# Dexcom Trend Settings
# ---------------------------
trend_descriptions = {
    1: 'Going up fast',
    2: 'Going up',
    3: 'Trending up',
    4: 'Steady',
    5: 'Trending down',
    6: 'Going down',
    7: 'Going down fast'
}
trend_unicode_arrows = {
    1: '↑↑',
    2: '↑',
    3: '↗',
    4: '→',
    5: '↘',
    6: '↓',
    7: '↓↓'
}

def trigger_home_assistant_alert(glucose_value, expected_bg, trend, timestamp):
    logging.info("Triggering Home Assistant alert...")
    webhook_url = "https://home.jadericdawson.com/api/webhook/glucose_alert"
    trend_description = trend_descriptions.get(trend, 'Unknown')
    payload = {
        "Current BG": glucose_value,
        "Expected BG": expected_bg,
        "Trend": trend_description,
        "Timestamp": timestamp.strftime('%H:%M')
    }
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            logging.info("Home Assistant alert triggered successfully.")
        else:
            logging.warning(f"Failed to trigger Home Assistant alert: {response.status_code}, {response.text}")
    except Exception as e:
        logging.error(f"Error triggering Home Assistant alert: {e}")

# ---------------------------
# Load Environment Variables and Dexcom Credentials
# ---------------------------
load_dotenv()
DEXCOM_USER = os.getenv("DEXCOM_USER")
DEXCOM_PASS = os.getenv("DEXCOM_PASS")
if not DEXCOM_USER or not DEXCOM_PASS:
    raise ValueError("Dexcom credentials not set in .env file.")

from pydexcom import Dexcom
dexcom = Dexcom(username=DEXCOM_USER, password=DEXCOM_PASS)

# ---------------------------
# Retrieve Dexcom Data and Save to CSV
# ---------------------------
max_minutes = 1440
glucose_data = dexcom.get_glucose_readings(minutes=max_minutes)
glucose_data = glucose_data[::-1]

timestamps = []
for reading in glucose_data:
    if reading.datetime.tzinfo is None:
        t = pytz.utc.localize(reading.datetime)
    else:
        t = reading.datetime.astimezone(pytz.utc)
    timestamps.append(t)
values = [r.value for r in glucose_data]
trends = [r.trend for r in glucose_data]

data_file = 'glucose_readings.csv'
def save_data(timestamp, value, trend, insulin=None, carbs=None):
    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S%z')
    data_entry = {'timestamp': timestamp_str, 'value': value, 'trend': trend, 'insulin': insulin, 'carbs': carbs}
    df = pd.DataFrame([data_entry])
    if not os.path.isfile(data_file):
        df.to_csv(data_file, index=False)
    else:
        df.to_csv(data_file, mode='a', header=False, index=False)

for t, v, tr in zip(timestamps, values, trends):
    save_data(t, v, tr)

def calculate_trend_line(timestamps, values, num_points=6):
    if len(values) < num_points:
        return None, None, None, None
    recent_timestamps = timestamps[-num_points:]
    recent_values = values[-num_points:]
    x = np.array([(t - recent_timestamps[0]).total_seconds()/60.0 for t in recent_timestamps])
    y = np.array(recent_values)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    trend_line = m * x + c
    expected_bg = y[-1] + m * 20
    return recent_timestamps, trend_line, m, expected_bg

# ---------------------------
# NEW: Helper Function to Add Rolling and Time Features
# ---------------------------
def add_rolling_and_time_features(df):
    """
    Given a DataFrame with a 'timestamp' and 'value' column,
    compute:
      - time_since_start (in seconds),
      - hourly_mean and hourly_std (rolling window of 12 readings),
      - cyclic time-of-day features: hour_sin, hour_cos,
      - cyclic day-of-week features: day_sin, day_cos.
    """
    # Ensure timestamp is datetime and in UTC
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    min_ts = df['timestamp'].min()
    df['time_since_start'] = df['timestamp'].apply(lambda x: (x - min_ts).total_seconds())
    
    # Rolling features: assume 12 readings ~1 hour
    df['hourly_mean'] = df['value'].rolling(window=12, min_periods=1).mean()
    df['hourly_std'] = df['value'].rolling(window=12, min_periods=1).std().fillna(0)
    
    # Time-of-day features (hour in 0-23)
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day-of-week features (0=Monday, 6=Sunday)
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

# ---------------------------
# Update: Get Future Predictions with Updated Features
# ---------------------------
def get_future_predictions(model, scaler, time_steps=5, future_steps=4):
    """
    Returns 4 predictions (each ~5 minutes apart, 20 minutes total) using updated rolling features.
    The features now include:
      ['value', 'trend', 'insulin', 'carbs', 'time_since_start', 'hourly_mean', 'hourly_std', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    """
    df_csv = pd.read_csv(data_file)
    df_csv = add_rolling_and_time_features(df_csv)
    
    # Ensure other columns are numeric
    df_csv['insulin'] = pd.to_numeric(df_csv['insulin'], errors='coerce').fillna(0)
    df_csv['carbs']   = pd.to_numeric(df_csv['carbs'], errors='coerce').fillna(0)
    df_csv['trend']   = pd.to_numeric(df_csv['trend'], errors='coerce').fillna(0)
    
    features = ['value', 'trend', 'insulin', 'carbs', 'time_since_start',
                'hourly_mean', 'hourly_std', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    data_model = df_csv[features]
    data_scaled = scaler.transform(data_model)
    sequence = data_scaled[-time_steps:]
    predictions = []
    for _ in range(future_steps):
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(seq_tensor)
        pred_val = pred.item()
        predictions.append(pred_val)
        last_row = sequence[-1].copy()
        last_row[0] = pred_val
        last_row[4] += 300  # Increase time_since_start by 300 seconds (5 minutes)
        sequence = np.vstack([sequence[1:], last_row])
    inv_predictions = []
    for p in predictions:
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = p
        inv = scaler.inverse_transform(dummy)[0, 0]
        inv_predictions.append(inv)
    return inv_predictions

# ---------------------------
# Define the Transformer Model
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.d_model = d_model
    def forward(self, src):
        src = src.permute(1, 0, 2)
        src = self.input_linear(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[-1]
        output = self.decoder(output)
        return output

# Our model was trained with 11 features.
input_size = 11
model = TransformerTimeSeries(input_size=input_size)

# ---------------------------
# Load Advanced Model & Scaler
# ---------------------------
try:
    advanced_model = torch.load("advanced_model.pt", map_location=torch.device('cpu'), weights_only=False)
    with open("advanced_scaler.pkl", "rb") as f:
        advanced_scaler = pickle.load(f)
    logging.info("Advanced model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading advanced model or scaler: {e}")
    advanced_model = None
    advanced_scaler = None

def update_predicted_bg():
    if advanced_model is not None and advanced_scaler is not None:
        try:
            future_preds = get_future_predictions(advanced_model, advanced_scaler, time_steps=5, future_steps=4)
            if future_preds:
                predicted_bg_var.set(f"Predicted BG in 5 mins: {future_preds[0]:.0f} mg/dL")
                logging.info(f"Predicted BG updated: {future_preds[0]:.0f} mg/dL")
            else:
                predicted_bg_var.set("Predicted BG: N/A")
        except Exception as e:
            logging.error(f"Error generating advanced model predictions: {e}")
            predicted_bg_var.set("Predicted BG: Error")
    else:
        predicted_bg_var.set("Predicted BG: Not available")

# ---------------------------
# GUI Setup with Tkinter
# ---------------------------
root = tk.Tk()
root.title("Glucose Monitor")

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

show_trend_arrows_var = tk.BooleanVar(value=True)
toggle_trend_arrows = ttk.Checkbutton(main_frame, text="Show Trend Arrows", variable=show_trend_arrows_var, command=lambda: create_plots())
toggle_trend_arrows.pack(side=tk.TOP, anchor='w', padx=5)

bg_frame = ttk.Frame(main_frame)
bg_frame.pack(side=tk.TOP, fill=tk.X, pady=5, padx=5)

current_bg_var = tk.StringVar(value=f"Current BG: {values[-1]} mg/dL")
current_trend_code = trends[-1]
current_trend_description = trend_descriptions.get(current_trend_code, "Unknown")
current_trend_var = tk.StringVar(value=f"Trend: {current_trend_description}")
expected_bg_var = tk.StringVar(value="Expected BG in 20 mins: Calculating...")
predicted_bg_var = tk.StringVar(value="Predicted BG in 5 mins: Calculating...")

bg_label = tk.Label(bg_frame, textvariable=current_bg_var, font=("Arial", 24), fg="blue")
bg_label.pack(side=tk.LEFT, padx=5)

trend_right_frame = ttk.Frame(bg_frame)
trend_right_frame.pack(side=tk.RIGHT, padx=5)
trend_label = tk.Label(trend_right_frame, textvariable=current_trend_var, font=("Arial", 18), fg="green")
trend_label.pack(anchor='w')
expected_bg_label = tk.Label(trend_right_frame, textvariable=expected_bg_var, font=("Arial", 16), fg="blue")
expected_bg_label.pack(anchor='w')
predicted_bg_label = tk.Label(trend_right_frame, textvariable=predicted_bg_var, font=("Arial", 16), fg="purple")
predicted_bg_label.pack(anchor='w')

notebook = ttk.Notebook(main_frame)
notebook.pack(expand=1, fill="both", padx=5, pady=5)

time_windows = [1, 3, 6, 12, 24, 'Max']
figures = {}
axes = {}
canvases = {}

for window in time_windows:
    tab = ttk.Frame(notebook)
    tab_text = f"{window}-Hour View" if window != 'Max' else "Max View"
    notebook.add(tab, text=tab_text)
    fig, ax = plt.subplots(figsize=(8, 5))
    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    figures[window] = fig
    axes[window] = ax
    canvases[window] = canvas

def init_plot(ax, title):
    ax.clear()
    ax.set_title(title)
    ax.set_xlabel("Timestamp (EDT)")
    ax.set_ylabel("Glucose Level (mg/dL)")
    ax.set_ylim(0, 400)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

local_tz = pytz.timezone('US/Eastern')
timestamps = [(t.astimezone(local_tz) if t.tzinfo else pytz.utc.localize(t).astimezone(local_tz)) for t in timestamps]

def create_plots():
    # Calculate a short linear trend from the most recent data (for labels)
    _, _, m, expected_bg = calculate_trend_line(timestamps, values, num_points=6)
    expected_bg_var.set(f"Expected BG in 20 mins: {int(expected_bg)} mg/dL")
    
    for window in time_windows:
        ax = axes[window]
        fig = figures[window]
        canvas = canvases[window]
        
        # Filter data for this window
        if window == 'Max':
            window_timestamps = timestamps
            window_values = values
            window_trends = trends
        else:
            now_local = datetime.datetime.now(local_tz)
            time_delta = datetime.timedelta(hours=window+1)
            cutoff_time = now_local - time_delta
            idx = [i for i, t in enumerate(timestamps) if t >= cutoff_time]
            window_timestamps = [timestamps[i] for i in idx]
            window_values = [values[i] for i in idx]
            window_trends = [trends[i] for i in idx]
        
        if not window_timestamps:
            logging.info(f"No data for {window}-hour window.")
            continue
        
        init_plot(ax, f"{window}-Hour Glucose Levels" if window != 'Max' else "Max Glucose Levels")
        ax.plot(window_timestamps, window_values, label='Glucose Level', color='blue')
        
        # Show trend arrows if toggled
        if show_trend_arrows_var.get():
            for i, trend_code in enumerate(window_trends):
                if trend_code in trend_unicode_arrows:
                    arrow = trend_unicode_arrows[trend_code]
                    ax.annotate(
                        arrow,
                        xy=(window_timestamps[i], window_values[i]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        color='blue'
                    )
        
        # Plot red linear trend, extended by 4 points (20 minutes)
        if len(window_timestamps) >= 10:
            x_dates, trend_line, slope, _ = calculate_trend_line(window_timestamps, window_values, num_points=6)
            if x_dates is not None:
                x_dates_local = [dt.astimezone(local_tz) for dt in x_dates]
                extended_x = list(x_dates_local)
                extended_y = list(trend_line)
                x_minutes = np.array([(dt - x_dates[0]).total_seconds()/60 for dt in x_dates_local])
                offset_last = x_minutes[-1]
                intercept = extended_y[-1] - slope*offset_last
                last_time = x_dates_local[-1]
                for i in range(1, 5):
                    offset_new = offset_last + i*5
                    future_time = last_time + datetime.timedelta(minutes=5*i)
                    extended_x.append(future_time)
                    extended_y.append(slope*offset_new + intercept)
                ax.plot(extended_x, extended_y, color="red", linestyle="--", linewidth=2, label="Linear Trend")
        
        # Plot green predicted line (model) for the next 4 points, continuous with the last data point
        if advanced_model is not None and advanced_scaler is not None:
            try:
                future_preds = get_future_predictions(advanced_model, advanced_scaler, time_steps=5, future_steps=4)
                last_data_timestamp = window_timestamps[-1]
                last_data_value = window_values[-1]
                predicted_timestamps = [last_data_timestamp + datetime.timedelta(minutes=5*(i+1)) for i in range(len(future_preds))]
                green_x = [last_data_timestamp] + predicted_timestamps
                green_y = [last_data_value] + future_preds
                ax.plot(green_x, green_y, color="green", linestyle="--", linewidth=2, label="Predicted")
                min_x = min(window_timestamps[0], green_x[0])
                max_x = max(window_timestamps[-1], green_x[-1])
                ax.set_xlim(min_x, max_x)
            except Exception as e:
                logging.error(f"Error generating advanced model predictions: {e}")
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=local_tz))
        fig.autofmt_xdate()
        ax.legend()
        canvas.draw()

create_plots()

def update_data():
    try:
        logging.info("Starting data update...")
        for attempt in range(5):
            try:
                glucose_reading = dexcom.get_current_glucose_reading()
                if glucose_reading:
                    logging.info(f"Glucose reading fetched on attempt {attempt+1}.")
                    break
            except requests.ConnectionError as conn_err:
                logging.warning(f"Connection error on attempt {attempt+1}: {conn_err}")
                sleep(10)
            except Exception as e:
                logging.error(f"Unexpected error on attempt {attempt+1}: {e}")
                sleep(10)
                continue
        else:
            logging.error("Failed to fetch glucose reading after multiple attempts.")
            root.after(60000, update_data)
            return
        
        if not glucose_reading:
            logging.warning("No new glucose reading available.")
            root.after(300000, update_data)
            return
        
        current_value = glucose_reading.value
        current_trend = glucose_reading.trend
        current_timestamp = glucose_reading.datetime.astimezone(local_tz)
        
        logging.info(f"New reading: BG={current_value}, Trend={current_trend}, Timestamp={current_timestamp}")
        timestamps.append(current_timestamp)
        values.append(current_value)
        trends.append(current_trend)
        save_data(current_timestamp, current_value, current_trend)
        logging.info("Data saved successfully.")
        
        current_bg_var.set(f"Current BG: {current_value} mg/dL")
        current_trend_var.set(f"Trend: {trend_descriptions.get(current_trend, 'Unknown')}")
        _, _, _, expected_bg = calculate_trend_line(timestamps, values, num_points=6)
        logging.info(f"Expected BG in 20 mins: {int(expected_bg)}")
        
        update_predicted_bg()  # Update predicted BG independently
        
        if current_value < 80 or current_value > 250:
            if expected_bg < 70 or expected_bg > 300:
                alert_message = (
                    f"ALERT! BG: {current_value} mg/dL, Expected BG: {int(expected_bg)} mg/dL, "
                    f"Trend: {trend_descriptions.get(current_trend, 'Unknown')}, Time: {current_timestamp.strftime('%H:%M')}"
                )
                logging.warning(alert_message)
                trigger_home_assistant_alert(current_value, int(expected_bg), current_trend, current_timestamp)
        
        create_plots()
        logging.info(f"Updated data: {current_value} mg/dL at {current_timestamp}")
        
    except RequestException as req_err:
        logging.error(f"Network error during update: {req_err}")
    except Exception as e:
        logging.error(f"Unexpected error during update: {e}")
    finally:
        logging.info("Scheduling next update in 5 minutes.")
        root.after(300000, update_data)

root.after(600, update_data)
root.mainloop()
