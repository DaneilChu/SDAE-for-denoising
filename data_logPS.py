#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:14:03 2017

@author: u3510120
Apply car noise to audio signal
Apply the new database
SNR=5
"""

from logfbank2 import power
import scipy.io.wavfile as wav
import numpy as np
import os
import glob
from keras.models import model_from_json


def addnoise(a,b,snr=5):
    path = os.getcwd()
    rate=[]
    data=[]
    data2 = []
    fn=[x for x in glob.glob(path+'/trainingdata/*.wav')][a:b]
    for x in fn:
        temprate, tempdata = wav.read(x)
        rate.append(temprate)
        data.append(tempdata)
    data = np.asanyarray(data)
    temprate, noise = wav.read('salamisound-1020082-street-noise-cars-in-both.wav')
    for i in range(data.shape[0]):
        n=np.random.randint(noise.shape[0]-data[i].shape[0])
        tempnoise=noise[n:n+data[i].shape[0],:]
        rms_noise = np.sqrt(np.mean(np.square(tempnoise.astype('int32'))))
        rms_signal = np.sqrt(np.mean(np.square(data[i].astype('int32'))))
        amp = rms_signal/(np.power(10,np.divide(snr,20,dtype='float64')))/rms_noise
        tempnoise = tempnoise*(amp**2)
        data2.append(tempnoise)
    data2 = np.asarray(data2)+ data
    validation_noise = data2[-1]
    validation_clean = data[-1]
    return data,data2,validation_clean,validation_noise,rate[0]

def loadmodel(wav_spectrogram):
    json_file = open('Autoencoder_logPS_estimate_noise.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder = model_from_json(loaded_model_json)
    # load weights into new model
    autoencoder.load_weights("Autoencoder_logPS_estimate_noise.h5")
    print("Loaded model from disk")
    wav_spectrogram_ps=[]
    for x in range(wav_spectrogram.shape[0]):
        wav_spectrogram_ps.append(autoencoder.predict(wav_spectrogram[x]))
    wav_spectrogram_ps=np.asarray(wav_spectrogram_ps)
    return wav_spectrogram_ps
    

def loaddata():
    if not os.path.exists('x_train_logPS.npy') & os.path.exists('x_train_noise_logPS.npy') & os.path.exists('x_test_logPS.npy') & os.path.exists('x_test_noise_logPS.npy'):
        readfile(6401,7200)
        readfile(7201,8000)
    x_train=np.load('x_train_logPS.npy')
    x_train_noise=np.load('x_train_noise_logPS.npy')
    x_test=np.load('x_test_logPS.npy')
    x_test_noise=np.load('x_test_noise_logPS.npy')
    x_validation_clean = np.load('x_validation_logPS.npy')
    x_validation_noise = np.load('x_validation_noise_logPS.npy')
    x_train_noise_ps=np.load('x_train_logPS_noise_ps.npy')
    x_test_noise_ps=np.load('x_test_logPS_noise_ps.npy')
    return x_train,x_train_noise,x_test,x_test_noise,x_validation_noise,x_validation_clean,x_train_noise_ps,x_test_noise_ps

def readfile(a,b):
    clean_speech,corrupted_speech,validation_clean,validation_noise,rate=addnoise(a,b)
    clean_coefficient = []
    print('calculating clean speech logfbank')
    for x in range(clean_speech.shape[0]):
        wav_spectrogram,phase = power(clean_speech[x], fft_size = 882,
                                    step_size = 441,
                                    spec_thresh = 0, 
                                    lowcut = 0, 
                                    highcut = 15000, 
                                    samplerate = rate,
                                    noise = False,
                                    log = True
                                    )
        clean_coefficient.append(wav_spectrogram)
    clean_coefficient = np.asarray(clean_coefficient)
    validation_clean,phase=power(validation_clean, fft_size = 882,
                                    step_size = 441,
                                    spec_thresh = 0, 
                                    lowcut = 0, 
                                    highcut = 15000, 
                                    samplerate = rate,
                                    noise = False,
                                    log = True
                                    )
    validation_noise,phase=power(validation_noise, fft_size = 882,
                                    step_size = 441,
                                    spec_thresh = 0, 
                                    lowcut = 0, 
                                    highcut = 15000, 
                                    samplerate = rate,
                                    noise = False,
                                    log = True
                                    )
    
    corrupted_coefficient = []
    'noise_ps=[]'
    print('calculating corrupted speech coefficient')
    for x in range(corrupted_speech.shape[0]):
        wav_spectrogram,phase=power(corrupted_speech[x], fft_size = 882,
                                    step_size = 441,
                                    spec_thresh = 0, 
                                    lowcut = 0, 
                                    highcut = 15000, 
                                    samplerate = rate,
                                    noise = False,
                                    log = True
                                    )
        'noise_ps.append(noise)'
        corrupted_coefficient.append(wav_spectrogram)
    
    corrupted_coefficient=np.asarray(corrupted_coefficient)
    noise_ps=loadmodel(corrupted_coefficient)
    'noise_ps=np.asarray(noise_ps)'

    
    print('Saving')
    clean_coefficient_train=clean_coefficient[0:clean_coefficient.shape[0]*8/10]
    corrupted_coefficient_train=corrupted_coefficient[0:corrupted_coefficient.shape[0]*8/10]
    clean_coefficient_test=clean_coefficient[clean_coefficient.shape[0]*8/10:clean_coefficient.shape[0]]
    corrupted_coefficient_test=corrupted_coefficient[corrupted_coefficient.shape[0]*8/10:corrupted_coefficient.shape[0]]
    noise_ps_train=noise_ps[0:noise_ps.shape[0]*8/10]
    noise_ps_test=noise_ps[noise_ps.shape[0]*8/10:noise_ps.shape[0]]

    if os.path.isfile('x_train_logPS.npy'):
        print('Next Round')
        clean_coefficient_train=np.concatenate((np.load('x_train_logPS.npy'),clean_coefficient_train),axis=0)
        corrupted_coefficient_train=np.concatenate((np.load('x_train_noise_logPS.npy'),corrupted_coefficient_train),axis=0)
        clean_coefficient_test=np.concatenate((np.load('x_test_logPS.npy'),clean_coefficient_test),axis=0)
        corrupted_coefficient_test=np.concatenate((np.load('x_test_noise_logPS.npy'),corrupted_coefficient_test),axis=0)
        noise_ps_train = np.concatenate((np.load('x_train_logPS_noise_ps.npy'),noise_ps_train),axis=0)
        noise_ps_test = np.concatenate((np.load('x_test_logPS_noise_ps.npy'),noise_ps_test),axis=0)
    np.save('x_train_logPS', clean_coefficient_train)
    np.save('x_train_noise_logPS', corrupted_coefficient_train)
    np.save('x_test_logPS', clean_coefficient_test)
    np.save('x_test_noise_logPS', corrupted_coefficient_test)
    np.save('x_train_logPS_noise_ps', noise_ps_train)
    np.save('x_test_logPS_noise_ps', noise_ps_test)
    np.save('x_validation_logPS',validation_clean)
    np.save('x_validation_noise_logPS',validation_noise)