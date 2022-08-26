# -*- coding: utf-8 -*-

from numba import jit, cuda

import random
import math
import time
import numpy as np
import mathutils
from random import randint
import sys
import pickle # so that dont have to train again while adjusting

## new additions
import pandas as pd
import pandas_datareader as web
import datetime as dt







#Commented temporarily need fix for tkinter DO NOT REMOVE - 10/03/21
#import tkinter as tk
#from tkinter import filedialog
#Commented temporarily need fix for tkinter DO NOT REMOVE - 10/03/21




import os
import numpy as np
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

## newly added lstm
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LSTM,BatchNormalization 
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv1D, MaxPooling1D
 


import matplotlib.pyplot as plt
import PIL
from pathlib import Path
import glob
import random
import time


import IPython
from IPython import *

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict,cross_val_score,cross_validate
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, r2_score




NAME = "discriminator-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))



######################################################## IMPORTS FINISHED ######################################################################################


############################################################

###########START DATE##############################################


startDate = "2011-05-18"
EPOCHS = 3

global counter

counter = 0








dates=pd.read_excel('C:/Users/h10up/Downloads/Hackathon_Dataset_NotIncludingTestSets__HITEN.xlsx')

df=pd.read_excel('C:/Users/h10up/Downloads/labels__revised_daysdataset_HACKATHON.xlsx')

DATASET_DIRECTORY = "C:/Users/h10up/Documents/DATASET_HACK/"


testin=pd.read_excel('C:/Users/h10up/Documents/TEST_HACKATHON__0001.xlsx')


## iloc to get position with [] and can use [:] slice



##print(df.iloc[0,0])

####df = df.apply(pd.to_datetime)


CATEGORIES = ["A_dir","B_dir","C_dir","D_dir","E_dir","F_dir","G_dir","H_dir","I_dir",
              "J_dir","K_dir","L_dir","M_dir","N_dir","O_dir","P_dir","Q_dir","R_dir",
              "S_dir","T_dir","U_dir","V_dir","W_dir","X_dir","Y_dir","Z_dir",
              "AA","AB","AC","AD","AE","AF","AG","AH","AI","AJ","AK","AL","AM","AN","AO","AP",
              "AQ","AR","AS","AT","AU","AV","AW","AX","AY","AZ","BA","BB","BC","BD","BE",
              "BF","BG","BH","BI","BJ","BK","BL","BM","BN","BO","BP","BQ","BR","BS",
              "BT","BU","BV","BW","BX","BY","BZ","CA","CB","CC","CD","CE","CF",
              "CG","CH"]

for category in CATEGORIES:
    path = os.path.join(DATASET_DIRECTORY, category)
    
    for fm_data in os.listdir(path):
        fm_df = pd.read_excel(os.path.join(path,fm_data))
        ##print(fm_df)
        
        break

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATASET_DIRECTORY, category)
        class_num = CATEGORIES.index(category)
    
        for fm_data in os.listdir(path):
    
            try:
                fm_df = pd.read_excel(os.path.join(path,fm_data))
                
                training_data.append([fm_df, class_num])
                
            
            except Exception as e:
                pass
            
                
create_training_data()

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)





X = np.array(X).reshape(86,28)



#########NORMALISE##############
X = np.array(X)
y=np.array(y)
)


xout=[]
yout=[]
Xin = []
Yin = []





xout = np.array(xout)

yout = np.array(yout)

X = np.round(X,1)

X, X_val, y, Y_val = train_test_split(X, y, test_size=0.2)
print(X.shape)
print(X_val.shape)
print(Y_val.shape)

X = np.reshape(X,(-1,28,1))


X = X/3858
X_val = X_val/3858


model = Sequential()

model.add(LSTM(128,input_shape=(28,1), activation='tanh', return_sequences=True))
##model.add(Dropout(0.2))
#model.add(BatchNormalization())



model.add(LSTM(128))
##model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(32,activation='tanh'))
model.add(Dropout(0.1))
#model.add(BatchNormalization())

model.add(Dense(10,activation='tanh'))
model.add(Dropout(0.1))
#model.add(BatchNormalization())

opt = tf.keras.optimizers.Adam(learning_rate=1e-2, decay=1e-2)


## mse
model.compile(loss='mse',
              optimizer=opt,
              metrics=['accuracy'])

## cs = EarlyStopping(monitor='loss', mode='min', verbose)


#print(X_val)

## epoch 6 lr 2 decay 3 best so far
model.fit(X,y, epochs=5,validation_data=(X_val,Y_val))

model.save('C:/Users/h10up/Documents/LSTM_MODEL_SAVED.model')

#print(testin)

print("testinshape")
print(testin.shape)




testin=np.array(testin/3858)

#print(testin.shape)

testin = np.reshape(testin,(1,28,1))



prediction = model.predict(testin[0])


prediction = np.reshape(prediction,(10,28))

print(prediction*3858)

prediction = np.array(prediction)

df_predict_out = pd.DataFrame(prediction*3858)

df_predict_out.to_excel("C:/Users/h10up/Documents/prediction_out.xlsx")
















# ### TRY KNN##############


# X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2)

# ### weights='distance'

# rgs = neighbors.KNeighborsRegressor(n_neighbors=68,weights='distance')




# predict = cross_val_predict(rgs, X, y, cv=5)


# #print(predict)

# print(mean_squared_error(y,predict))
# print(r2_score(y,predict))


# error = []

# for k in range(1,51):
#     knn= neighbors.KNeighborsRegressor(n_neighbors=k)
#     y_pred = cross_val_predict(knn, X,y, cv=5)
#     error.append(mean_squared_error(y, y_pred))


# plt.plot(range(1,51),error)


# # accuracy = clf.score(X_test, y_test)
# # print(accuracy)



# # score_train = mean_squared_error()

























