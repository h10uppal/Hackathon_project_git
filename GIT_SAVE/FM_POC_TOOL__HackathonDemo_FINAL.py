# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import xlrd

from openpyxl import *


import random
import math
import time
import numpy as np
import mathutils
from random import randint
import sys

import pandas as pd
import pandas_datareader as web
import datetime as dt

import os

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








def getTestInputPath():
    
    print("TEST")
    
    testin_path = filedialog.askopenfilename()
    
    global testin
    
    testin=pd.read_excel(testin_path)
    
    
    testin=np.array(testin/3858)
    

    
    testin = np.reshape(testin,(-1,28,1))
    
    print(testin)
    
    #return testin
    
  
    
def getOutputPath():
    
    global outPath
    
    outPath = filedialog.askopenfilename()
    
    print(outPath)
    
    
    #return outPath
    
  

# def runModel(model_FM,testin,outPath):

#     prediction = model_FM.predict(testin)
    
#     prediction = np.array(prediction*3858)
    
#     print(prediction)

#     df_predict_out = pd.DataFrame(prediction)
    
#     df_predict_out.to_excel(outPath)



        

def getModelPath():
    

    model_path = filedialog.askdirectory()
    
    
    global model_FM
    
    model_FM = tf.keras.models.load_model(model_path)
    
    



def runModel():
    
    prediction = model_FM.predict(testin)
    
    #prediction = np.reshape(prediction,(-1,28))
    
    prediction = prediction*3858
    prediction = np.array(prediction)
    
    print(prediction)

    df_predict_out = pd.DataFrame(prediction)
    
    df_predict_out.to_excel(outPath)

    
    

      
        


def main():

  
    root= tk.Tk()
    
    canvas1 = tk.Canvas(root, width = 500, height = 400)
    canvas1.pack()
    
    
    
    


    
    getTestInputButton = tk.Button(root,text='Get Test Input Path (.xlsx)', command=lambda: getTestInputPath(), bg='dark blue', fg='white', font=('helvetica', 8, 'bold'))
    canvas1.create_window(250, 150, window=getTestInputButton)
    
    
    out_Path = tk.Button(root,text='Output Path (.xlsx)', command=lambda: getOutputPath(), bg='dark blue', fg='white', font=('helvetica', 8, 'bold'))
    canvas1.create_window(250, 200, window=out_Path)
    
    
    getModelButton = tk.Button(root,text='Get Model Path (.model folder)', command=lambda: getModelPath(), bg='dark blue', fg='white', font=('helvetica', 8, 'bold'))
    canvas1.create_window(250, 100, window=getModelButton)
    
    
    
    run_Model = tk.Button(root,text='Run Model', command=lambda: runModel(), bg='dark blue', fg='white', font=('helvetica', 8, 'bold'))
    canvas1.create_window(250, 275, window=run_Model)
    

    
    root.mainloop()


main()      