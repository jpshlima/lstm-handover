# -*- coding: utf-8 -*-
"""

@author: jpshlima
"""

# let's try to run LSTM on RSRP values from ns-3 dual strip simulation

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# First, read and prepare RSRP data
file = 'DlRsrpSinrStats_hom-0_ttt-64.17.txt'
h = open(file, 'r')
hlines = h.readlines()

base = []
for line in hlines:
    base.append(line.split())
# Organize data frame
base = pd.DataFrame(base)
#base.head(n=5)
base.drop(columns=[1, 3, 6, 7], inplace=True)
#base.drop(columns=[3, 6, 7], inplace=True)
base.columns=['time', 'IMSI', 'rsrp', 'sinr']
#base.columns=['time', 'cellID', 'IMSI', 'rsrp', 'sinr']
base = base.iloc[1:]
# transform RSRP from linear to dB
base['rsrp'] = np.log10(base['rsrp'].values.astype(float))*10
#mybase = base.loc[base['IMSI'].astype(int)==18]
# get only RSRP values from 1 UE as time series
myrsrp = []
myrsrp = base.loc[base['IMSI'].astype(int)==12, 'rsrp']
myrsrp.reset_index(drop=True, inplace=True)
myrsrp = pd.DataFrame(myrsrp).values

# apply MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
myrsrp_norm = scaler.fit_transform(myrsrp)
# train and test split 
rsrptrain = myrsrp_norm[0:8820, :]
rsrptest = myrsrp_norm[8821:9800, :]

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

# more layers
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

# final layer
regressor.add(Dense(units = 1, activation = 'linear'))

# compiling
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
history = regressor.fit(prev, real_rsrp, epochs = 100, batch_size = 32, validation_split=0.1)


# testing phase
# preparing inputs for test
inputs = myrsrp_norm[len(myrsrp_norm) - len(rsrptest) - 100:]
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

# get real RSRP test values to plot and compare
real_rsrp_test = myrsrp[8821:9800, :]
mae = mean_absolute_error(real_rsrp_test, prediction)

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
plt.show()



# saving LSTM neural network
regressor_json = regressor.to_json()
with open('lstm_rsrp.json', 'w') as json_file:
    json_file.write(regressor_json)
regressor.save_weights('lstm_rsrp.h5')
    

# saving history variable from Keras training
history_df = pd.DataFrame(history.history)
with open('history.csv', mode='w') as f:
    history_df.to_csv(f, index=False)