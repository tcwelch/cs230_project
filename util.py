import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import regularizers
import tensorflow as tf

def normalize(X):
	m = np.shape(X)[0] # number of examples
	n_H = np.shape(X)[1] # number of features in an example 
	n_W = np.shape(X)[2]
	n_C = np.shape(X)[3]
	mu = np.reshape(np.sum(X,axis=0),(1,n_H,n_W,n_C))/m
	return (X - mu)/255

def neural_network(X,Y,layer_dims,epochs = 500,batch_size = 25,loss='binary_crossentropy',lambd = 1e-4):
	""" 
	Inputs: X,Y,layer_dims,epochs, batch_size, loss
	Outputs: model

	"""
	m = np.shape(X)[0] 
	n = np.shape(X)[1] 
	layers = len(layer_dims)

	model = Sequential()
	model.add(Dense(layer_dims[0], input_dim = n, activation = 'relu',kernel_initializer = 'glorot_uniform',bias_initializer='zeros',kernel_regularizer=regularizers.l2(lambd), bias_regularizer=regularizers.l2(lambd)))

	for i in range(1,layers - 1):
		neurons = layer_dims[i]
		model.add(Dense(neurons, activation='relu',kernel_initializer = 'glorot_uniform',bias_initializer='zeros', kernel_regularizer=regularizers.l2(lambd), bias_regularizer=regularizers.l2(lambd)))

	model.add(Dense(layer_dims[-1], activation='softmax'))
	model.compile(loss = tf.keras.losses.CategoricalCrossentropy(), optimizer = 'adam', metrics=['accuracy'])
	model.fit(X, Y, epochs = epochs, batch_size = batch_size)

	return model