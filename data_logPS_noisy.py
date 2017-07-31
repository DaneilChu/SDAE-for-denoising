#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat March 25 15:14:03 2017

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
    data2 = np.asarray(data2)
    data3 = data2+ data
    validation_input = data3[-1]
    validation_output = data2[-1]
    return data2,data3,validation_output,validation_input,rate[0]

def loaddata(phase=False):
    if not os.path.exists('x_train_output_logPS_estimate_noise.npy') & os.path.exists('x_train_input_logPS_estimate_noise.npy') & os.path.exists('x_test_output_logPS_estimate_noise.npy') & os.path.exists('x_test_input_logPS_estimate_noise.npy'):
        readfile(1601,2400)
        readfile(2401,3200)
    x_train_output=np.load('x_train_output_logPS_estimate_noise.npy')
    x_train_input=np.load('x_train_input_logPS_estimate_noise.npy')
    x_test_output=np.load('x_test_output_logPS_estimate_noise.npy')
    x_test_input=np.load('x_test_input_logPS_estimate_noise.npy')
    x_validation_output = np.load('x_validation_output_logPS_estimate_noise.npy')
    x_validation_input = np.load('x_validation_input_logPS_estimate_noise.npy')
    if phase:
        x_validation_output_phase = np.load('x_validation_output_phase_logPS_estimate_noise.npy')
        x_validation_input_phase = np.load('x_validation_input_phase_logPS_estimate_noise.npy')
        return x_train_output,x_train_input,x_test_output,x_test_input,x_validation_input,x_validation_output,x_validation_input_phase,x_validation_output_phase
    else:
        return x_train_output,x_train_input,x_test_output,x_test_input,x_validation_input,x_validation_output


def readfile(a,b):
    output_speech,input_speech,validation_output,validation_input,rate=addnoise(a,b)
    output_coefficient = []
    print('calculating output speech logfbank')
    for x in range(output_speech.shape[0]):
        wav_spectrogram,phase = power(output_speech[x], fft_size = 882,
                                    step_size = 441,
                                    spec_thresh = 0, 
                                    lowcut = 0, 
                                    highcut = 15000, 
                                    samplerate = rate,
                                    noise = False,
                                    log = True
                                    )
        output_coefficient.append(wav_spectrogram)
    output_coefficient = np.asarray(output_coefficient)
    validation_output,validation_output_phase=power(validation_output, fft_size = 882,
                                    step_size = 441,
                                    spec_thresh = 0, 
                                    lowcut = 0, 
                                    highcut = 15000, 
                                    samplerate = rate,
                                    noise = False,
                                    log = True
                                    )
    validation_input,validation_input_phase=power(validation_input, fft_size = 882,
                                    step_size = 441,
                                    spec_thresh = 0, 
                                    lowcut = 0, 
                                    highcut = 15000, 
                                    samplerate = rate,
                                    noise = False,
                                    log = True
                                    )
    
    input_coefficient = []
    print('calculating input speech coefficient')
    for x in range(input_speech.shape[0]):
        wav_spectrogram,phase=power(input_speech[x], fft_size = 882,
                                    step_size = 441,
                                    spec_thresh = 0, 
                                    lowcut = 0, 
                                    highcut = 15000, 
                                    samplerate = rate,
                                    noise = False,
                                    log = True
                                    )
        input_coefficient.append(wav_spectrogram)
    input_coefficient=np.asarray(input_coefficient)
    
    print('Saving')
    output_coefficient_train=output_coefficient[0:output_coefficient.shape[0]*8/10]
    input_coefficient_train=input_coefficient[0:input_coefficient.shape[0]*8/10]
    output_coefficient_test=output_coefficient[output_coefficient.shape[0]*8/10:output_coefficient.shape[0]]
    input_coefficient_test=input_coefficient[input_coefficient.shape[0]*8/10:input_coefficient.shape[0]]

    if os.path.isfile('x_train_input_logPS_estimate_noise.npy'):
        print('Next Round')
        output_coefficient_train=np.concatenate((np.load('x_train_output_logPS_estimate_noise.npy'),output_coefficient_train),axis=0)
        input_coefficient_train=np.concatenate((np.load('x_train_input_logPS_estimate_noise.npy'),input_coefficient_train),axis=0)
        output_coefficient_test=np.concatenate((np.load('x_test_output_logPS_estimate_noise.npy'),output_coefficient_test),axis=0)
        input_coefficient_test=np.concatenate((np.load('x_test_input_logPS_estimate_noise.npy'),input_coefficient_test),axis=0)

    np.save('x_train_output_logPS_estimate_noise', output_coefficient_train)
    np.save('x_train_input_logPS_estimate_noise', input_coefficient_train)
    np.save('x_test_output_logPS_estimate_noise', output_coefficient_test)
    np.save('x_test_input_logPS_estimate_noise', input_coefficient_test)
    np.save('x_validation_output_logPS_estimate_noise',validation_output)
    np.save('x_validation_input_logPS_estimate_noise',validation_input)
    np.save('x_validation_input_phase_logPS_estimate_noise',validation_input_phase)
    np.save('x_validation_output_phase_logPS_estimate_noise',validation_output_phase)

    