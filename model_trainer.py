import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime

# ---------------------------
# 1. Data Loading and Preprocessing
# ---------------------------
data_file = "glucose_readings.csv"
df = pd.read_csv(data_file)

# Ensure the 'timestamp' column is in datetime format (using utc=True for consistency)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)

# Fill missing numeric values for insulin and carbs with 0
df['insulin'] = pd.to_numeric(df['insulin'], errors='coerce').fillna(0)
df['carbs']   = pd.to_numeric(df['carbs'], errors='coerce').fillna(0)

# Ensure 'trend' is numeric
df['trend'] = pd.to_numeric(df['trend'], errors='coerce').fillna(0)

# Create a new feature: time since the first reading (in seconds)
min_timestamp = df['timestamp'].min()
df['time_since_start'] = df['timestamp'].apply(lambda x: (x - min_timestamp).total_seconds())

# --- NEW: Compute additional long-term features ---
# Assuming readings are roughly every 5 minutes, 12 readings ≈ 1 hour.
df['hourly_mean'] = df['value'].rolling(window=12, min_periods=1).mean()
df['hourly_std'] = df['value'].rolling(window=12, min_periods=1).std().fillna(0)

# --- NEW: Add Time-of-Day and Day-of-Week Features ---
# Extract hour (0-23) and day-of-week (0=Monday, 6=Sunday)
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Encode cyclically: hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
# Encode cyclically: day-of-week
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Select features for training; target will be the next 'value'
features = ['value', 'trend', 'insulin', 'carbs', 'time_since_start', 
            'hourly_mean', 'hourly_std', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
data = df[features]

# Normalize features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# ---------------------------
# 2. Create Rolling Sequences
# ---------------------------
def create_sequences(data, time_steps=5):
    """
    Creates sequences of length `time_steps` with the target being the next reading's 'value'.
    Assumes the first column of data corresponds to the glucose value.
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        seq = data[i:i + time_steps]
        target = data[i + time_steps, 0]  # target is the glucose value
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

time_steps = 5  # Using the last 5 readings to predict the next
X, y = create_sequences(data_scaled, time_steps)

# Split the data into training and validation sets (80/20 split, preserving time order)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# ---------------------------
# 3. Define the Transformer Model
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0)]

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1):
        """
        input_size: Number of input features.
        d_model: Dimension for linear projection.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Dimension of the feedforward network.
        dropout: Dropout rate.
        """
        super(TransformerTimeSeries, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)  # Predict a single value
        self.d_model = d_model
        
    def forward(self, src):
        # src shape: (batch_size, seq_len, input_size)
        src = src.permute(1, 0, 2)  # -> (seq_len, batch_size, input_size)
        src = self.input_linear(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[-1]  # Use the output from the last time step
        output = self.decoder(output)
        return output

# Update input_size to 11 (we now have 11 features)
input_size = len(features)
model = TransformerTimeSeries(input_size=input_size)

# ---------------------------
# 4. Train the Model
# ---------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

# ---------------------------
# 5. Save the Trained Model and Scaler
# ---------------------------
torch.save(model, "advanced_model.pt")
with open("advanced_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training complete. Model and scaler have been saved.")
