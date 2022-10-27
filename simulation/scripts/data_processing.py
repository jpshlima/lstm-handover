# -*- coding: utf-8 -*-
"""

@author: jpshlima
"""

# here we load many txt output files from ns-3, load the trained LSTM,
# apply the prediction, and create the database for the second stage: classification

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from keras.models import model_from_json
import os
import re
import math



def readingHOlogs(ho_files):
    #for hofile in ho_files:
    e = open(ho_files, 'r')
    hlines = e.readlines()
    ho_log = []
    for hline in hlines:
        hline = re.sub(':', '', hline) #removing ':' from 2nd column
        ho_log.append(hline.split())      
    # List into dataframe (DF)
    ho_log = pd.DataFrame(ho_log)
    ho_log = ho_log.apply(pd.to_numeric, errors='coerce')
    ho_log = ho_log.dropna()
    ho_log.drop(columns=[1, 3, 4], inplace=True)
    # Assign names to DF columns
    ho_log.columns = ['time', 'IMSI']
    ho_log.reset_index(drop=True, inplace=True)
    #ho_logs['time'] = ho_logs['time'].astype(float)
    ho_log['ho_trig'] = 1
    return ho_log


def readingRSRPlogs(file_names):
    base = []
    # reading txt files containing rsrp data
    #for file in file_names:
    h = open(file_names, 'r')
    next(h)
    hlines = h.readlines()
    for line in hlines:
        base.append(line.split())
    base = pd.DataFrame(base)
    #base.head(n=5) 
    base.drop(columns=[1, 3, 5, 6], inplace=True)
    base.columns=['time', 'IMSI', 'rsrp']
    base.reset_index(drop=True, inplace=True)
    # transform RSRP from linear to dB
    base['rsrp'] = np.log10(base['rsrp'].values.astype(float))*10
    #base.head()
    base['ho_trig'] = '0'
    base['time'] = base['time'].astype(float).round(3)
    base['IMSI'] = base['IMSI'].astype(int)
    #base = base.apply(pd.to_numeric, errors='coerce')
    return base

def concatLogs(base, ho_logs):
    result = pd.concat([base, ho_logs], ignore_index=True, sort=False)
    #result['time'] = result['time'].astype(float)
    result.sort_values(by=['IMSI', 'time'], inplace=True)
    result.reset_index(drop=True, inplace=True)
    result['rsrp'] = result['rsrp'].astype(float)
#    rsrp = result['rsrp']
#    #rsrp.reset_index(drop=True, inplace=True)
    for i in range (2, result.shape[0]-1):
        if (math.isnan(result['rsrp'][i]) == True):
            result['rsrp'][i] = (result['rsrp'][i-1]+result['rsrp'][i+1])*0.5
#    result['rsrp'] = rsrp
    result.reset_index(drop=True, inplace=True)
    return (result)
    

def loadTrainedNN():
    # loading saved parameters 
    saved_net = open('lstm_rsrp.json', 'r')
    struct_net = saved_net.read()
    saved_net.close()
    regressor = model_from_json(struct_net)
    regressor.load_weights('lstm_rsrp.h5')
    return (regressor)


def getPredictions(base):
    prevs = []
    # for loop to fill out the database. getting 1 UE at a time
    for j in range(1, 21):
        # get only RSRP values from 1 UE as time series
        myrsrp = []
        myrsrp = base.loc[base['IMSI'].astype(int)==j, 'rsrp']
        myrsrp.reset_index(drop=True, inplace=True)
        myrsrp = pd.DataFrame(myrsrp).values
        time = []
        time = base.loc[base['IMSI'].astype(int)==j, 'time']
        time.reset_index(drop=True, inplace=True)
        time = time.loc[200:9800]
        time = np.array(time)
        ho_trig = base.loc[base['IMSI'].astype(int)==j, 'ho_trig']
        ho_trig.reset_index(drop=True, inplace=True)
        ho_trig = ho_trig.loc[200:9800]
        ho_trig = np.array(ho_trig)
        # apply MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        myrsrp_norm = scaler.fit_transform(myrsrp)
        # train and test split 
        #rsrptrain = myrsrp_norm[0:8820, :]
        rsrptest = myrsrp_norm[200:9800, :]
        
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
        prediction = []
        prediction = regressor.predict(x_test)
        # undo normalization for better viewing our results
        prediction = scaler.inverse_transform(prediction)
        aux = np.zeros((prediction.size, 4))
    #        prediction = np.hstack((prediction, np.full((prediction.shape[0],1), j)))
    #        prediction = np.vstack(prediction)
        for i in range (prediction.size):
            aux[i, 0] = prediction[i]
            aux[i, 1] = time[i]
            aux[i, 2] = j
            aux[i, 3] = ho_trig[i]
        prevs.append(aux)
    prevs = np.vstack(prevs)
    return (prevs)



# To read .txt HO logs
ho_files = ['ho_log19.txt', 'ho_log14.txt', 'ho_log17.txt', 'ho_log25.txt', 'ho_log28.txt', 'ho_log34.txt']

# first let's read 2 or more txt files and group them into 1 dataframe
file_names = ['DlRsrpSinrStats_hom-0_ttt-64.19.txt', 'DlRsrpSinrStats_hom-0_ttt-64.14.txt', 'DlRsrpSinrStats_hom-0_ttt-64.17.txt', 'DlRsrpSinrStats_hom-0_ttt-64.25.txt', 'DlRsrpSinrStats_hom-0_ttt-64.28.txt', 'DlRsrpSinrStats_hom-0_ttt-64.34.txt']#, 'DlRsrpSinrStats_hom-1_ttt-40.txt']

# load trained LSTM
regressor = loadTrainedNN()
concatbases = []

for j in range (len(file_names)):
    base = []
    base = readingRSRPlogs(file_names[j])
    ho_logs = []
    ho_logs = readingHOlogs(ho_files[j])
    # concatening dataframes
    base = concatLogs(base, ho_logs)
    # variable for getting predictions out of LSTM
    prevs = getPredictions(base)
        
    # loop for filling variable
    x_test = []
    for i in range (50, prevs.shape[0]):
        x_test.append(prevs[i-50:i, 0])
    x_test = np.array(x_test)        
    
    classification_base = pd.DataFrame(x_test)
    classification_base['label'] = prevs[49:prevs.shape[0]-1, 3]
    concatbases.append(classification_base)
#    from collections import Counter
#    counter = Counter(classification_base['label'])
#    print(counter)

concatbases = np.vstack(concatbases)
concatbases2 = pd.DataFrame(concatbases)
concatbases2.to_csv('concatbases.csv', index=False)
#classification_base.to_csv('classification_base.csv', index=False)
