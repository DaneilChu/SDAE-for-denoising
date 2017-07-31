#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed March 25 16:38:06 2017

@author: u3510120
"""

import data_logPS_noisy
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from keras.optimizers import SGD
from keras import backend as K
from keras.models import model_from_json
import os
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.layers.advanced_activations import LeakyReLU

path=os.getcwd()
fft_size = 882
step_size = 441
rate=44100
keeptrain = bool(int(raw_input("Keep Train?: ")))
epoch=int(raw_input("Please enter epoch: "))
learn = float(raw_input("Please enter learn: "))
mon=float(raw_input("Please enter mon: "))
batch = 512
loss_function = "mse"

leaky_coefficient=0.3



#Loading Data

if keeptrain==True:
    json_file = open('Autoencoder_logPS_estimate_noise.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder = model_from_json(loaded_model_json)
    # load weights into new model
    autoencoder.load_weights("Autoencoder_logPS_estimate_noise.h5")
    print("Loaded model from disk")
else:
    w_coefficient = float(raw_input("Please enter w_coefficient: "))
    w_contraint = float(raw_input("Please enter w_contraint: "))
    print('Fine-tuning:')
    
    act=LeakyReLU(leaky_coefficient)
    autoencoder = Sequential()
    autoencoder.add(Dense(350, input_shape=(442,),W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(act)
    autoencoder.add(Dense(240, W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(act)
    autoencoder.add(Dense(180, W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(act)
    autoencoder.add(Dense(120, W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(act)
    autoencoder.add(Dense(80, W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(act)
    autoencoder.add(Dense(120,activation='linear', W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(Dense(180,activation='linear', W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(Dense(240,activation='linear', W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(Dense(350,activation='linear', W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(Dense(442,activation='linear', W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    



op = SGD(lr=learn,momentum=mon)

def dist(y_true, y_pred):
    return 10*np.log10(K.mean(np.abs(y_true - y_pred)))

def reduct(x_true,y_pred):
    return 10*np.log10(K.mean(np.abs(x_true - y_pred)))

autoencoder.compile(optimizer=op, loss='mse',metrics=[dist,reduct])
for x in range(epoch/50):
    x_train,x_train_noise,x_test,x_test_noise,x_validation_noise,x_validation_clean= data_logPS_noisy.loaddata()
    x_train=np.concatenate(x_train).astype(None)
    x_train_noise=np.concatenate(x_train_noise).astype(None)
    x_test=np.concatenate(x_test).astype(None)
    x_test_noise=np.concatenate(x_test_noise).astype(None)
        
    
    
    autoencoder.fit(x_train_noise,
                    x_train,
                    nb_epoch=50,
                    shuffle=True,
                    batch_size=batch,
                    verbose=1,
                    validation_data=(x_test_noise,x_test)
                    )
        # serialize model to JSON
    model_json = autoencoder.to_json()
    with open("Autoencoder_logPS_estimate_noise.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    autoencoder.save_weights("Autoencoder_logPS_estimate_noise.h5",overwrite=True)
    print("Saved model to disk")

model_config = str(autoencoder.get_config()) + "\n"
result = "result: " + str(autoencoder.evaluate(x_test_noise, x_test[:x_test.shape[0]],batch_size=128))+"\n"
text =model_config+result
with open("Autoencoder_logPS_estimate_noise.txt", "a") as text_file:
    text_file.write(text)

