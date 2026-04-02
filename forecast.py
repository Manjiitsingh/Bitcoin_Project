import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout

# 1. LOAD DATA
df = pd.read_csv('bitcoin.xlsx - Sheet1.csv')
data = df['Close'].values.reshape(-1, 1)

# 2. PREPARE SCALER
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. RE-BUILD & TRAIN (Same as before)
prediction_days = 60
x_train, y_train = [], []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
print("AI is training one last time...")
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

# 4. PREDICT TOMORROW
# We take the VERY LAST 60 days from your data to predict the next day
real_data = [scaled_data[len(scaled_data) - prediction_days:len(scaled_data), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

print("-" * 30)
print(f"PREDICTION FOR TOMORROW: ${prediction[0][0]:.2f}")
print("-" * 30)