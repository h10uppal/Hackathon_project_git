# -*- coding: utf-8 -*-

from numba import jit, cuda

import random
import math
import time
import numpy as np
import mathutils
from random import randint
import sys
import pickle 

## new additions
import pandas as pd
import pandas_datareader as web
import datetime as dt



import xlsxwriter

import os
import numpy as np
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

## newly added lstm
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LSTM
from tensorflow.keras.callbacks import TensorBoard


import matplotlib.pyplot as plt
import PIL
from pathlib import Path
import glob
import random
import time


import IPython
from IPython import *



NAME = "discriminator-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))



######################################################## IMPORTS FINISHED ######################################################################################

###########START DATE##############################################
startDate = "2011-05-18"
EPOCHS = 3

global counter

counter = 0
##################### START DATE#######################################





#################MODELS_DEFINITIONS#########################################################################

def discrim_model_second():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(32, 1, strides=1, padding='same',
                                     input_shape=X.shape[1:]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.35))

    model.add(layers.Conv1D(64,1, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.35))
    
    model.add(layers.Conv1D(64,1, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.35))
    
    model.add(layers.Conv1D(32,1, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.35))
    

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    
    model.save('C:/Users/h10up/Documents/h10_hackathon_discriminator.model')

    # model.compile(loss="binary_crossentropy",
    #                 optimizer = "adam",
    #                 metrics=['accuracy'])


    ##model.fit(X,y,batch_size=6, epochs=EPOCHS,validation_split=0.3, callbacks=[tensorboard])

    print("in discrim")
    
    return model



def make_generator_model():

    model = tf.keras.Sequential()
    
    ##### Change from 7,7,256 to 8,8,30
    
    
    ### may need to change use_bias to true########
    model.add(layers.Dense(29*1, use_bias=False,input_shape=(29,1)))
    
    ###model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(None)))
    
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    print(X.shape[1:])
    ### Change from 7,7,256 to 8,8,30
    # model.add(layers.Reshape(-1,29))
    # print(model.output_shape)
    # assert model.output_shape == (None, 29, 1)  # Note: None is the batch size


    ###### Changed all Conv2DTranspose padding from 'same' to 'valid' now getting assertion error look line 382 #############################


    model.add(layers.Conv1DTranspose(128, 2, strides=2, padding='same', use_bias=False))
    print(model.output_shape)
    assert model.output_shape == (None, 58, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    
    ### Might need to add upsample layer then pooling then can add another conv2dtranspose
    

    model.add(layers.Conv1DTranspose(64, 2, strides=2, padding='same', use_bias=False))
    print(model.output_shape)
    assert model.output_shape == (None, 116, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(1, 2, strides=2, padding='same', use_bias=False, activation='tanh'))
    print(model.output_shape)
    assert model.output_shape == (None, 232, 1)
    
    
    
    model.save('C:/Users/h10up/Documents/h10_hackathon_generator.model')


    return model


#################CREATING__TRAINING__DATA#########################################################################



dates=pd.read_excel('C:/Users/h10up/Downloads/Hackathon_Dataset_NotIncludingTestSets__HITEN.xlsx')

df=pd.read_excel('C:/Users/h10up/Downloads/labels__revised_daysdataset_HACKATHON.xlsx')

DATASET_DIRECTORY = "C:/Users/h10up/Documents/HACKATHON_DATASETDIR"


## iloc to get position with [] and can use [:] slice



##print(df.iloc[0,0])

####df = df.apply(pd.to_datetime)


CATEGORIES = ["data"]

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





X = np.array(X).reshape(-1,29,1)

pickle_out = open("C:/Users/h10up/Documents/Hackathon_X.pickle.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("C:/Users/h10up/Documents/Hackathon_y.pickle.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()




X = pickle.load(open("C:/Users/h10up/Documents/Hackathon_X.pickle.pickle","rb"))
y = pickle.load(open("C:/Users/h10up/Documents/Hackathon_y.pickle.pickle","rb"))

##print(X)

X = np.array(X/3800)

X = np.array(X)
y=np.array(y)





print(X.shape[1:])

#print(training_data)


generator = make_generator_model()

noise = tf.random.normal([29,1])

print(noise.shape)

generated_random = generator(noise, training=False)

generated_random = np.array(generated_random).reshape(-1,29,1)




discriminator = discrim_model_second()

decision = discriminator(generated_random)

#print(decision)





############LOSS OPTIMIZERS##############################################


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)





def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss



def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)



generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)





# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)



@tf.function
def train_step(images):
    
    noise = tf.random.normal([29,1])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      
      real_output = discriminator(images, training=True) 
      
      ##print(real_output)
      

      
      generated_images = tf.reshape(generated_images,(-1,29,1,1))

      
      fake_output = discriminator(generated_images, training=True) 
      gen_loss = generator_loss(fake_output) 
      disc_loss = discriminator_loss(real_output, fake_output) 
      gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  
    


def train(dataset, epochs):
    

    
    for epoch in range(epochs):
        start = time.time()
        
        for image_batch in dataset:
          

            seed = tf.random.normal([29,1])
             
            ## WORKING    
            train_step(image_batch)
            
            

            ##predictons = model(seed,training=False)
            

    
            # print("TEST 0") 
            generate_and_save_images(generator,
                                      epoch + 1,
                                      seed)
            # print("TEST 0") 
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start)) 




def generate_and_save_images(model, epoch, test_input):
    
    predictions = model(test_input, training=False)
    prediction_less = tf.reshape(predictions,(-1,29,1))
    prediction_less_nonormal = prediction_less*3800
    ##prediction_less_nonormal = prediction_less
    if epoch == final_epoch:
        
        #print(prediction_less_nonormal)
        
        out_nump=np.array(prediction_less_nonormal)
        
        print(out_nump.shape)
        
        out_reshp = np.reshape(out_nump,(8,29))
        
        print(out_reshp)
        
        df_out = pd.DataFrame(out_reshp)
        
        df_out.to_excel("C:/Users/h10up/Documents/data_out.xlsx")
        
    

    
   
    
    

# 300
final_epoch = 350

train(X[None,], final_epoch)






####tensor_data = tf.convert_to_tensor(df,np.float32)
























