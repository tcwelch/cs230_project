import data_proc 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import regularizers
import tensorflow as tf


def neural_network(X,Y,nn_dims,epochs = 500,batch_size = 25,loss='binary_crossentropy',lambd = 1e-4):
	""" 
	Inputs:
		 X - training data (np.array(rows: #examples; cols: #features))
		 Y - training data labels (np.array(rows: #examples; cols: 1))
		 nn_dims - list with number of neurons in each layer [#input layer, # 1st hidden layer, ..., 1 neuron with sigmoid]
		 epochs - number of iterations to run gradient descent 
		 batch_size - number of training examples to include in a batch
		 loss - loss function for gradient descent to optimize over (input as string)

	Outputs:
		trained_model - model trained in keras

	"""
	m = np.shape(X)[0] # number of examples
	n = np.shape(X)[1] # number of features in an example 
	layers = len(nn_dims) # number of layers

	model = Sequential()
	model.add(Dense(nn_dims[0], input_dim = n, activation = 'relu',kernel_initializer = 'glorot_uniform',bias_initializer='zeros',kernel_regularizer=regularizers.l2(lambd), bias_regularizer=regularizers.l2(lambd)))

	for i in range(1,layers - 1):
		neurons = nn_dims[i]
		model.add(Dense(neurons, activation='relu',kernel_initializer = 'glorot_uniform',bias_initializer='zeros', kernel_regularizer=regularizers.l2(lambd), bias_regularizer=regularizers.l2(lambd)))

	model.add(Dense(6, activation='softmax'))
	print(model.summary())
	model.compile(loss = loss, optimizer = 'adam', metrics=['accuracy'])
	model.fit(X, Y, epochs = epochs, batch_size = batch_size)

	return model


def main():
	#------ Training ------
	X_train, y_train = data_proc.get_batch('archive/one-indexed-files-notrash_train.txt',224,224)
	m = X_train.shape[0]

	# Run training data through RestNet50
	transfer_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
	A_transfer = transfer_model.predict(X_train) # sized at (m,7,7,2048)
	A_transfer = np.reshape(A_transfer,(m,-1)) # sized at (m,7*7*2048)
	print("A_transfer.shape")
	print(A_transfer.shape)

	# Train our model on output of Resnet50
	our_model = neural_network(A_transfer, y_train,[1024,256,64,16,6],epochs = 5)

	# Prediction on training data
	y_train_pred = our_model.predict(A_transfer)

	#------ Validation ------
	# Run validation data through RestNet50
	X_valid, y_valid = data_proc.get_batch('archive/one-indexed-files-notrash_val.txt',224,224)
	m = X_test.shape[0]
	A_transfer = transfer_model.predict(X_valid) # sized at (m,7,7,2048)
	A_transfer = np.reshape(A_transfer,(m,-1)) # sized at (m,7*7*2048)

	# Prediction on validation data
	y_valid_pred = our_model.predict(A_transfer)



	print("y_train_pred.shape")
	print(y_train_pred.shape)
	print("y_train_pred[1,:]")
	print(y_train_pred[1,:])
	print("np.sum(y_train_pred[1,:])")
	print(np.sum(y_train_pred[1,:]))


if __name__ == "__main__":
    main()
