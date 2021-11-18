import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq, irfft, fft, fftfreq, ifft
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model

class ecgTools:
	def __init__(self, sampleRate=0, duration=0):
		self.sampleRate, self.duration = sampleRate, duration
	
	# Takes in the name/location of a ecg reading and outputs a numpy array of its contents
	def getRawData(self,filename):
		fIn = open(filename,"r")
		
		ecgRaw = fIn.readlines()
		rawLen = len(ecgRaw)
		for i in range(rawLen):
			ecgRaw[i] = float(ecgRaw[i])
		ecgNp = np.array(ecgRaw)
		return ecgNp
	
	# Takes in the record, physionet database name, start point and end point then returns
	# a numpy array of the record from physionet
	def getPhysionetData(self, record, path, start, end):
		sig = wfdb.rdrecord(record,
							channels=[0],
							sampfrom = start,
							sampto = end,
							return_res = 32,
							pn_dir = path)
		sig = sig.__dict__['p_signal']
		sig = np.array(sig)
		return sig
	
	# Takes in an ecg numpy array and returns a version that is the correct shape for the 
	# model 
	def reshape(self, ecgIn):
		return ecgIn.reshape(1,ecgIn.shape[0],1)
		
	# Takes in ecg array and applys a butterworth filter to the data and returns the result
	def butterFilt(self,ecgIn):
		T = 20
		fs = 250
		cutoff = 30
		nyq = 0.5*fs
		
		order = 2
		n=int(T*fs)
		
		normalCutoff = cutoff/nyq
		b, a = butter(order, normalCutoff, btype = 'low',analog=False)
		y = filtfilt(b,a,ecgIn)
		return y
		
	# Takes in an ecg array and a window size and applys a Simple Moving Average filter
	# using the given window size to the array then returns the resulting array
	def sma(self, ecgIn, window):
		out = [None]*len(ecgIn)
		for i in range(len(ecgIn)):
			out[i] = np.sum(ecgIn[i:i+window])/window
		return out
		
	# Takes in ecg array, window size, and the smoothing rate and applys an
	# Exponential Moving Average with the given window size and smoothing rate then returns the
	# final array
	def ema(self, ecgIn, window, smoothing):
		out = [np.sum(ecgIn[:window])/window]
		for i in ecgIn[window:]:
			out.append((i * (smoothing/(1+window))) + out[-1] * (1-(smoothing/(1+window))))
		return out
		
	# Takes in an ECG array and returns the fft transfromation and frequency 
	def doFFT(self, ecgIn):
		N = self.sampleRate*self.duration
		transformed = rfft(ecgIn)
		freq = rfftfreq(N,1/self.sampleRate)
		return transformed, freq
		
	# Takes in the fft transformation and frequency arrays from doFFT as well as a target Hz and
	# zeros out the fft transformation in a range of +/- 10 indecies from the target Hz then
	# returns the same FFT transfrom as passed in with the target area around the target Hz
	# zeroed out
	def denoiseFFT(self, FFT, freq, targetHz):
		pointsPerFreq = len(freq)/(self.sampleRate/2)
		targetIndex = int(pointsPerFreq*targetHz)
		FFT[targetIndex-(10*int(pointsPerFreq)):targetIndex+(10*int(pointsPerFreq))] = 0
		return FFT
	
	# Method that takes in an ecg array and a target Hz and uses the doFFT and denoiseFFT methods
	# in conjunction to produce a denoised version of the original ecg input array
	def fftFilt(self, ecgIn, targetHz):
		fft, freq = self.doFFT(ecgIn)
		denoised = self.denoiseFFT(fft, freq, targetHz)
		return irfft(denoised)
		
	# Takes in a list of ecg recordings, a start and end point, and a list of labels in the
	# same order as the ecg's then plots them on a single graph
	def ecgPlot(self, ecgList, start, end, labelList):
		for i in range(len(ecgList)):
			plt.plot(ecgList[i][start:end], label = labelList[i])
		plt.legend(loc = "upper left")
		plt.show()
	
	def fftPlot(self, ecgIn, pltLabel):
		fft, freq = self.doFFT(ecgIn)
		plt.figure(100)
		plt.plot(freq,np.abs(fft))
		plt.title(f'{pltLabel} FFT')
		plt.figure(200)
		denoised = self.denoiseFFT(fft, freq, 70)
		plt.plot(freq,np.abs(denoised))
		plt.show()
	
	# Given a properly shaped ecg array and the file path for a model ecgPredict will
	# provide a prediction on what kind of rhythm the ecg is as a string
	def predict(self, ecgIn, modelLocation):
		
		categories = ["Normal Sinus Rhythm","Congestive Heart Failure","Atrial Fibrillation"]

		model_tf = tf.keras.models.Sequential()
		d = 0.6
		model_tf.add(layers.Conv1D(filters=32,
								   kernel_size=(1,),
								   activation=tf.keras.layers.LeakyReLU(alpha=0.01),
								   input_shape=(5000,1,)))
		model_tf.add(layers.Conv1D(32,3,activation='relu'))
		model_tf.add(layers.Conv1D(32,3,activation='relu'))
		model_tf.add(layers.Dropout(d))

		model_tf.add(layers.GlobalAveragePooling1D())

		model_tf.add(layers.Dense(64,activation='relu'))
		model_tf.add(layers.Dropout(d))

		model_tf.add(layers.Dense(3,activation='softmax'))

		model_tf.load_weights(modelLocation)
		
		prediction = model_tf.predict(ecgIn)
		for i in range(len(prediction)):
			prediction[i] = np.round(prediction[i],decimals=1)
		return categories[np.argmax(prediction,axis=None,out=None)]
