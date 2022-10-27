# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 18:48:25 2022

@author: jpshlima
"""

# load best anatel lstm model, load base for predictions and test classifiers

from keras.models import model_from_json
def loadTrainedNN():
    # loading saved parameters 
    saved_net = open('anatel_lstm_rsrp.json', 'r')
    struct_net = saved_net.read()
    saved_net.close()
    regressor = model_from_json(struct_net)
    regressor.load_weights('anatel_lstm_rsrp.h5')
    return (regressor)


def getPredictions(df):
    prevs = []
    # get only RSRP values from 1 UE as time series
    rsrp = df['RSRP'].values
    rsrp = rsrp.reshape(-1,1)
    from sklearn.preprocessing import MinMaxScaler
    # apply MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    rsrp_norm = scaler.fit_transform(rsrp)
    # train and test split 
    rsrptest = rsrp_norm[8006:8896, :]
    ho_trig = df['ho_trig'].values
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
    aux = np.zeros((prediction.size, 2))
    #        prediction = np.hstack((prediction, np.full((prediction.shape[0],1), j)))
    #        prediction = np.vstack(prediction)
    for i in range (prediction.size):
        aux[i, 0] = prediction[i]
        #aux[i, 1] = time[i]
        #aux[i, 2] = j
        aux[i, 1] = ho_trig[i]
    prevs.append(aux)
    prevs = np.vstack(prevs)
    return (prevs)


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



regressor = loadTrainedNN()

prevs = getPredictions(df)

concatbases = []
# loop for filling variable
x_test = []
for i in range (50, prevs.shape[0]):
    x_test.append(prevs[i-50:i, 0])
x_test = np.array(x_test)        

classification_base = pd.DataFrame(x_test)
classification_base['label'] = prevs[49:prevs.shape[0]-1, 1]
concatbases.append(classification_base)
concatbases = np.vstack(concatbases)
concatbases2 = pd.DataFrame(concatbases)
concatbases2.to_csv('anatel_concatbases.csv', index=False)


