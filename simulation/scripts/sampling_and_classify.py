# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:06:06 2022

@author: jpshlima
"""

# load CSV database and apply over/undersampling techniques and classification

import numpy as np
import pandas as pd

# read CSV
#df = pd.read_csv('concatbases.csv')
# get variables for label and previsores
#label = df['10'].values
#inputs = df.loc[:, df.columns != '10'].values


#from collections import Counter
#counter = Counter(label)
#print(counter)

#nova = df[df['50']==1]
#shuffle = df[df['50'] != 1].sample(n = 2000)
#nova2 = pd.concat([nova, shuffle], ignore_index=True, sort=False)
#
#label = nova2['50'].values
#inputs = nova2.loc[:, df.columns != '50'].values
#
#
#from imblearn.under_sampling import TomekLinks
#tomek = TomekLinks()
#inputs, label = tomek.fit_resample(inputs, label)
#
##from collections import Counter
##counter = Counter(label)
##print(counter)
##
#from imblearn.over_sampling import SMOTE
#smote = SMOTE()
#
#inputs, label = smote.fit_resample(inputs, label)
#counter = Counter(label)
#print(counter)

#
## save new database for classification
#classifybase = pd.DataFrame(inputs)
#classifybase['label'] = label
#classifybase.to_csv('classifybase.csv', index=False)


# read CSV
df = pd.read_csv('classifybase.csv')
# get variables for label and previsores
label = df['label'].values
inputs = df.loc[:, df.columns != 'label'].values

# applies standardization to inputs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)




##applies k-Fold, classifies, repeat the process X times and calculate means and std.dev
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
#from sklearn.svm import SVC
## Aplica o StratifiedKFold
#resultados33 = []
#f1score33 = []
#for i in range(10):
#    kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = i)
#    resultados1 = []
#    matriz1 = []
#    f1score = []
#    for n_train, n_test in kfold.split(inputs, np.zeros(shape=(inputs.shape[0], 1))):
#        svm = SVC(kernel = 'rbf', C = 500.0, )
#        svm.fit(inputs[n_train], label[n_train])
#        previsoes = svm.predict(inputs[n_test])
#        precisao = accuracy_score(label[n_test], previsoes)
#        matriz1.append(confusion_matrix(label[n_test], previsoes))
#        resultados1.append(precisao)
#        f1score1 = f1_score(label[n_test], previsoes)
#        f1score.append(f1score1)
#    resultados1 = np.asarray(resultados1)
#    media = resultados1.mean()
#    resultados33.append(media)
#    f1score33.append(f1score)
#    matriz_final = np.mean(matriz1, axis = 0)
#resultados33 = np.asarray(resultados33)
#f1score = np.mean(f1score)
#
#resultados33.mean()
#resultados33.std()
#matriz_final


##applies k-Fold, classifies, repeat the process X times and calculate means and std.dev
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
#from sklearn.neural_network import MLPClassifier
## Aplica o StratifiedKFold
#resultados33 = []
#f1score33 = []
#for i in range(10):
#    kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = i)
#    resultados1 = []
#    matriz1 = []
#    f1score = []
#    for n_train, n_test in kfold.split(inputs, np.zeros(shape=(inputs.shape[0], 1))):
#        model = MLPClassifier(solver='lbfgs', max_iter=2000, hidden_layer_sizes=[120,120])
#        model.fit(inputs[n_train], label[n_train])
#        previsoes = model.predict(inputs[n_test])
#        precisao = accuracy_score(label[n_test], previsoes)
#        matriz1.append(confusion_matrix(label[n_test], previsoes))
#        resultados1.append(precisao)
#        f1score1 = f1_score(label[n_test], previsoes)
#        f1score.append(f1score1)
#    resultados1 = np.asarray(resultados1)
#    media = resultados1.mean()
#    resultados33.append(media)
#    f1score33.append(f1score)
#    matriz_final = np.mean(matriz1, axis = 0)
#resultados33 = np.asarray(resultados33)
#f1score = np.mean(f1score33)
#
#resultados33.mean()
#resultados33.std()
#matriz_final


##applies k-Fold, classifies, repeat the process X times and calculate means and std.dev
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
#from sklearn.neighbors import KNeighborsClassifier
## Aplica o StratifiedKFold
#resultados33 = []
#f1score33 = []
#for i in range(10):
#    kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = i)
#    resultados1 = []
#    matriz1 = []
#    f1score = []
#    for n_train, n_test in kfold.split(inputs, np.zeros(shape=(inputs.shape[0], 1))):
#        model = KNeighborsClassifier(n_neighbors=2, p=1, algorithm='ball_tree', leaf_size=200)
#        model.fit(inputs[n_train], label[n_train])
#        previsoes = model.predict(inputs[n_test])
#        precisao = accuracy_score(label[n_test], previsoes)
#        matriz1.append(confusion_matrix(label[n_test], previsoes))
#        resultados1.append(precisao)
#        f1score1 = f1_score(label[n_test], previsoes)
#        f1score.append(f1score1)
#    resultados1 = np.asarray(resultados1)
#    media = resultados1.mean()
#    resultados33.append(media)
#    f1score33.append(f1score)
#    matriz_final = np.mean(matriz1, axis = 0)
#resultados33 = np.asarray(resultados33)
#f1score = np.mean(f1score33)
#
#resultados33.mean()
#resultados33.std()
#matriz_final


#applies k-Fold, classifies, repeat the process X times and calculate means and std.dev
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
# Aplica o StratifiedKFold
resultados33 = []
f1score33 = []
for i in range(10):
    kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = i)
    resultados1 = []
    matriz1 = []
    f1score = []
    for n_train, n_test in kfold.split(inputs, np.zeros(shape=(inputs.shape[0], 1))):
        model = RandomForestClassifier(n_estimators=200)
        model.fit(inputs[n_train], label[n_train])
        previsoes = model.predict(inputs[n_test])
        precisao = accuracy_score(label[n_test], previsoes)
        matriz1.append(confusion_matrix(label[n_test], previsoes))
        resultados1.append(precisao)
        f1score1 = f1_score(label[n_test], previsoes)
        f1score.append(f1score1)
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean()
    resultados33.append(media)
    f1score33.append(f1score)
    matriz_final = np.mean(matriz1, axis = 0)
resultados33 = np.asarray(resultados33)
f1score = np.mean(f1score33)

resultados33.mean()
resultados33.std()
matriz_final


##applies k-Fold, classifies, repeat the process X times and calculate means and std.dev
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
#from sklearn.linear_model import LogisticRegression
## Aplica o StratifiedKFold
#resultados33 = []
#f1score33 = []
#for i in range(1):
#    kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = i)
#    resultados1 = []
#    matriz1 = []
#    f1score = []
#    for n_train, n_test in kfold.split(inputs, np.zeros(shape=(inputs.shape[0], 1))):
#        model = LogisticRegression(C=200, solver='liblinear', max_iter=1500)
#        model.fit(inputs[n_train], label[n_train])
#        previsoes = model.predict(inputs[n_test])
#        precisao = accuracy_score(label[n_test], previsoes)
#        matriz1.append(confusion_matrix(label[n_test], previsoes))
#        resultados1.append(precisao)
#        f1score1 = f1_score(label[n_test], previsoes)
#        f1score.append(f1score1)
#    resultados1 = np.asarray(resultados1)
#    media = resultados1.mean()
#    resultados33.append(media)
#    f1score33.append(f1score)
#    matriz_final = np.mean(matriz1, axis = 0)
#resultados33 = np.asarray(resultados33)
#f1score = np.mean(f1score33)
#
#resultados33.mean()
#resultados33.std()
#matriz_final






