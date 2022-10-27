# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 19:12:15 2022

@author: jpshlima
"""

# try to predict RSRP from Anatel samples. Load LSTM and test


# loading saved LSTM network and testing in different RSRP time series

from keras.models import model_from_json

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# loading saved parameters 
saved_net = open('lstm_rsrp.json', 'r')
struct_net = saved_net.read()
saved_net.close()

regressor = model_from_json(struct_net)
regressor.load_weights('lstm_rsrp.h5')


# First, read and prepare RSRP data
files = ['export_tim_14.csv', 'export_tim_18.csv', 'export_tim_19.csv']
df = pd.concat((pd.read_csv(f) for f in files))
df.drop(df.columns[[0,1,2,4,5,7,8,9,10]], axis=1, inplace=True)
df['ho_trig'] = 0
df.reset_index(drop=True, inplace=True)

for i in range (2, df.shape[0]-1):
    if ((df['PCI'][i] != df['PCI'][i-1]) == True):
        df['ho_trig'][i] = 1



#print(sum(df['ho_trig']))

# get only RSRP values from 1 UE as time series
rsrp = df['RSRP'].values
rsrp = rsrp.reshape(-1,1)
# apply MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
rsrp_norm = scaler.fit_transform(rsrp)
# train and test split 
rsrptrain = rsrp_norm[0:3999, :]
rsrptest = rsrp_norm[4000:8896, :]

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

real_rsrp_test = rsrp[4000:8896, :]
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

