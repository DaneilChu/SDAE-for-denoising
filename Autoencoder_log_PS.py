#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:38:06 2017

@author: u3510120
"""

import data_logPS
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
import math

path=os.getcwd()
fft_size = 882
step_size = 441
rate=44100
keeptrain = bool(int(raw_input("Keep Train?: ")))
epoch=int(raw_input("Please enter epoch: "))
p = int(raw_input("Please enter the proportion of training: "))
learn = float(raw_input("Please enter learn: "))
mon=float(raw_input("Please enter mon: "))
batch = 512
loss_function = "mse"

leaky_coefficient=0.01



#Loading Data

if keeptrain==True:
    json_file = open('Autoencoder_log_PS.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder = model_from_json(loaded_model_json)
    # load weights into new model
    autoencoder.load_weights("Autoencoder_log_PS.h5")
    print("Loaded model from disk")
else:
    w_coefficient = float(raw_input("Please enter w_coefficient: "))
    w_contraint = float(raw_input("Please enter w_contraint: "))
    print('Fine-tuning:')
    
    act=LeakyReLU(leaky_coefficient)
    autoencoder = Sequential()
    autoencoder.add(Dense(1024, input_shape=(1768,),W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(act)
    autoencoder.add(Dense(512, W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(act)
    autoencoder.add(Dense(300, W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(act)
    autoencoder.add(Dense(180, W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(act)
    autoencoder.add(Dense(120, W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(act)
    autoencoder.add(Dense(180,activation='linear', W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(Dense(300,activation='linear', W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(Dense(512,activation='linear', W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(Dense(1024,activation='linear', W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(Dense(1768,activation='linear', W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    autoencoder.add(Dense(442,activation='linear', W_regularizer=l2(w_coefficient),W_constraint = maxnorm(w_contraint)))
    



op = SGD(lr=learn,momentum=mon)

def ath(nfft,rate):
    feq_base=float(rate/nfft)
    feq=[]
    for x in range(nfft/2+1):
        feq.append(feq_base*x)
    feq=np.asarray(feq)
    threshold=[]
    for x in range(nfft/2+1):
        if x ==0:
            num=3.64*math.pow(feq[3]/1000,-0.8)-6.5*math.exp(-0.6*math.pow(feq[3]/1000-3.3,2))+0.001*math.pow(feq[3]/1000,4)
            threshold.append(num)
        else:
            num=3.64*math.pow(feq[x]/1000,-0.8)-6.5*math.exp(-0.6*math.pow(feq[x]/1000-3.3,2))+0.001*math.pow(feq[x]/1000,4)
            threshold.append(num)
    threshold=np.asarray(threshold)
    threshold=threshold+np.abs(np.min(threshold))+1
    weight=1/threshold
    return weight

weight=ath(fft_size,rate)

def custom_objective(y_true, y_pred):
    cce=K.sum(K.square(K.abs(weight*(y_pred - y_true))),axis=-1)
    return cce

def dist(y_true, y_pred):
    return 10*np.log10(K.mean(np.abs(y_true - y_pred)))

def reduct(x_true,y_pred):
    return 10*np.log10(K.mean(np.abs(x_true - y_pred)))

autoencoder.compile(optimizer=op, loss='mse',metrics=[dist,reduct])
for x in range(epoch/50):
    x_train,x_train_noise,x_test,x_test_noise,x_validation_noise,x_validation_clean,x_train_noise_ps,x_test_noise_ps= data_logPS.loaddata()
    n_train = np.random.randint(x_train.shape[0]*0.65)
    n_test = np.random.randint(x_test.shape[0]*0.65)
    x_train=np.concatenate(x_train[n_train:n_train+x_train.shape[0]/3]).astype(None)
    x_train_noise=np.concatenate(x_train_noise[n_train:n_train+x_train_noise.shape[0]/3]).astype(None)
    x_test=np.concatenate(x_test[n_test:n_test+x_test.shape[0]/3]).astype(None)
    x_test_noise=np.concatenate(x_test_noise[n_test:n_test+x_test_noise.shape[0]/3]).astype(None)
    x_train_noise_ps=np.concatenate(x_train_noise_ps[n_train:n_train+x_train_noise_ps.shape[0]/3]).astype(None)
    x_test_noise_ps=np.concatenate(x_test_noise_ps[n_test:n_test+x_test_noise_ps.shape[0]/3]).astype(None)
        
    
    autoencoder.fit(np.concatenate((np.concatenate((np.zeros([1,442]),x_train_noise[0:-1,:]),axis=0),x_train_noise[:x_train_noise.shape[0]/p],np.concatenate((x_train_noise[1:,:],np.zeros([1,442])),axis=0),x_train_noise_ps),axis=1)
                    , x_train[:x_train.shape[0]/p],
                    nb_epoch=50,
                    shuffle=True,
                    batch_size=batch,
                    verbose=1,
                    validation_data=(np.concatenate((np.concatenate((np.zeros([1,442]),x_test_noise[0:-1,:]),axis=0),x_test_noise[:x_test_noise.shape[0]/p],np.concatenate((x_test_noise[1:,:],np.zeros([1,442
                                                     ])),axis=0),x_test_noise_ps),axis=1)
                                      , x_test[:x_test.shape[0]/p])
                    )
        # serialize model to JSON
    model_json = autoencoder.to_json()
    with open("Autoencoder_log_PS.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    autoencoder.save_weights("Autoencoder_log_PS.h5",overwrite=True)
    print("Saved model to disk")

txt =",fft size: " + str(fft_size) + ", step_size: " + str(step_size) + "\n" + "batch_size: " + str(batch) +   ", learning rate: " + str(learn) + ", Momentum: " + str(mon) + ", Loss_function: "+loss_function+", leaky_coefficient:"+str(leaky_coefficient)+",w_contraint:"+str(w_contraint)+",w_coefficient:"+str(w_coefficient)+"\n"                                   
model_config = str(autoencoder.get_config()) + "\n"
result = "result: " + str(autoencoder.evaluate(np.concatenate((np.concatenate((x_test_noise[-1:,:],x_test_noise[0:-1,:]),axis=0),x_test_noise[:x_test_noise.shape[0]/p],np.concatenate((x_test_noise[1:,:],x_test_noise[0,:].reshape(1,x_test_noise.shape[1])),axis=0),x_test_noise),axis=1), x_test[:x_test.shape[0]/p],batch_size=128))+"\n"
text =txt+model_config+result
with open("Autoencoder_log_PS.txt", "a") as text_file:
    text_file.write(text)
