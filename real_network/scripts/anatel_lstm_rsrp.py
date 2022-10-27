# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 08:27:48 2022

@author: jpshlima
"""

# test LSTM model with Anatel data. train and test LSTM

import numpy as np
import pandas as pd
# First, read and prepare RSRP data
files = ['export_tim_14.csv', 'export_tim_18.csv', 'export_tim_19.csv']
df = pd.concat((pd.read_csv(f) for f in files))
df.drop(df.columns[[0,1,2,4,5,7,8,9,10]], axis=1, inplace=True)
df['ho_trig'] = 0
df.reset_index(drop=True, inplace=True)

for i in range (2, df.shape[0]-1):
    if ((df['PCI'][i] != df['PCI'][i-1]) == True):
        df['ho_trig'][i] = 1


# get only RSRP values from 1 UE as time series
rsrp = df['RSRP'].values
rsrp = rsrp.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
# apply MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
rsrp_norm = scaler.fit_transform(rsrp)
# train and test split 
rsrptrain = rsrp_norm[0:8005, :]
rsrptest = rsrp_norm[8006:8896, :]

# Training phase
# initialing variables
prev = []
real_rsrp = []

# filling for 100-sample prediction
for i in range(100, rsrptrain.size):
    prev.append(rsrptrain[i-100:i, 0])
    real_rsrp.append(rsrptrain[i, 0])

# adapting formats (only 1 dimension)
prev, real_rsrp = np.array(prev), np.array(real_rsrp)
prev = np.reshape(prev, (prev.shape[0], prev.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
# starting regressor
regressor = Sequential()
regressor.add(LSTM(units = 120, return_sequences = True, input_shape = (prev.shape[1], 1)))
# using dropout to avoid overfitting
regressor.add(Dropout(0.3))

# more layers
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

# more layers
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

## more layers
#regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.3))

# more layers
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))


# final layer
regressor.add(Dense(units = 1, activation = 'linear'))

# compiling
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
history = regressor.fit(prev, real_rsrp, epochs = 100, batch_size = 128, validation_split=0.1)


# testing phase
# preparing inputs for test
inputs = rsrp_norm[len(rsrp_norm) - len(rsrptest) - 100:]
inputs = inputs.reshape(-1, 1)
#inputs = scaler.transform(inputs)

# loop for filling variable
x_test = []
for i in range (100, inputs.size):
    x_test.append(inputs[i-100:i, 0])
# format adapting
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction = regressor.predict(x_test)
# undo normalization for better viewing our results
prediction = scaler.inverse_transform(prediction)

from sklearn.metrics import mean_absolute_error
# check MAE
real_rsrp_test = rsrp[8006:8896, :]
mae = mean_absolute_error(real_rsrp_test, prediction)

import matplotlib.pyplot as plt
# visualization
plt.plot(real_rsrp_test, color = 'red', label = 'Real RSRP')
plt.plot(prediction, color = 'blue', label = 'Prediction')
plt.title('RSRP values prediction')
plt.xlabel('Time (samples)')
plt.ylabel('RSRP (dB)')
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation Loss'], loc='upper left')
plt.grid()
plt.show()


## saving LSTM neural network
#regressor_json = regressor.to_json()
#with open('anatel_lstm_rsrp.json', 'w') as json_file:
#    json_file.write(regressor_json)
#regressor.save_weights('anatel_lstm_rsrp.h5')
#    
#
## saving history variable from Keras training
#history_df = pd.DataFrame(history.history)
#with open('anatel_history.csv', mode='w') as f:
#    history_df.to_csv(f, index=False)
