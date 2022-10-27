# -*- coding: utf-8 -*-
"""

@author: jpshlima
"""

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

# prepare database for other testing

# First, read and prepare RSRP data
file = 'DlRsrpSinrStats_hom-0_ttt-64.txt'
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
myrsrp = base.loc[base['IMSI'].astype(int)==4, 'rsrp']
myrsrp.reset_index(drop=True, inplace=True)
myrsrp = pd.DataFrame(myrsrp).values



# apply MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
myrsrp_norm = scaler.fit_transform(myrsrp)
# train and test split 
rsrptrain = myrsrp_norm[0:8820, :]
rsrptest = myrsrp_norm[8821:9800, :]


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


