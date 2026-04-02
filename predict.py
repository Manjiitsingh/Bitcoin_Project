""""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. LOAD DATA
# This reads your CSV file
df = pd.read_csv('bitcoin.xlsx - Sheet1.csv')
data = df['Close'].values.reshape(-1, 1)

# 2. SCALE DATA
# AI models work best when numbers are small (between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. CREATE SEQUENCES
# We tell the model to look at the past 60 days to predict the next day
prediction_days = 60
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 4. BUILD THE LSTM AI MODEL
model = Sequential()
# The "Brain" layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2)) # Prevents the model from just memorizing (overfitting)
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Final output: The predicted price

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. TRAIN THE MODEL
# This is where the computer "learns" from your data
print("Training started... please wait.")
model.fit(x_train, y_train, epochs=10, batch_size=32)

# 6. VISUALIZE THE RESULTS
# This part creates a graph to show you how the model did
predicted_prices = model.predict(x_train)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_train.reshape(-1, 1))

plt.plot(actual_prices, color='black', label='Actual Price')
plt.plot(predicted_prices, color='green', label='AI Prediction')
plt.title('Bitcoin Price Prediction')
plt.legend()
plt.show()
"""

"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. LOAD THE FULL DATA
# Ensure your file is named exactly 'bitcoin.xlsx - Sheet1.csv' in the folder
df = pd.read_csv('bitcoin.xlsx - Sheet1.csv')
data = df['Close'].values.reshape(-1, 1)

# 2. SCALE DATA
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. CREATE SEQUENCES (The 60-day memory)
prediction_days = 60 
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 4. BUILD THE MODEL (Advanced Architecture)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2)) # Prevents the AI from "cheating" by memorizing
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. TRAIN THE MODEL
print("Training on full data... This may take 1-2 minutes.")
# Epochs = 25 means the AI will read the whole history 25 times to learn
model.fit(x_train, y_train, epochs=25, batch_size=32)

# 6. PREDICT AND GRAPH
predictions = model.predict(x_train)
predictions = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_train.reshape(-1, 1))

plt.figure(figsize=(12,6))
plt.plot(actual_prices, color='black', label='Actual Bitcoin Price')
plt.plot(predictions, color='green', label='AI Prediction')
plt.title('Bitcoin Price Prediction (Full History)')
plt.xlabel('Time (Days)')
plt.ylabel('Price in USD')
plt.legend()

print("Process Complete. Opening Graph...")
plt.show()
"""
import matplotlib
matplotlib.use('TkAgg') # FIX FOR PROBLEM 1 (Graph Window)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. LOAD DATA
df = pd.read_csv('bitcoin.xlsx - Sheet1.csv')
data = df['Close'].values.reshape(-1, 1)

# 2. SCALE
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. SEQUENCES (60 days)
prediction_days = 60 
x_train, y_train = [], []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 4. FIXED BRAIN (Stronger Layers)
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=False)) # Increased units to 100
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. TRAIN LONGER
print("Training the smarter model... Wait for Epoch 50.")
model.fit(x_train, y_train, epochs=50, batch_size=32) # FIX FOR PROBLEM 2 (Accuracy)

# 6. GRAPH
predictions = model.predict(x_train)
predictions = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_train.reshape(-1, 1))

plt.figure(figsize=(12,6))
plt.plot(actual_prices, color='black', label='Actual Bitcoin Price')
plt.plot(predictions, color='green', label='AI Prediction')
plt.title('Bitcoin Price Prediction - FIXED')
plt.legend()
plt.show()