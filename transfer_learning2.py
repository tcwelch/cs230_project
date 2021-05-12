import data_proc 
import util
import numpy as np
import tensorflow as tf
import confusion_matrix


def main():
	#------ Training ------
	X_train, y_train = data_proc.get_batch('archive/one-indexed-files-notrash_train.txt',224,224)
	m = X_train.shape[0]

	# Run training data through RestNet50
	transfer_model = tf.keras.applications.DenseNet121(include_top=False,weights="imagenet")
	A_transfer = transfer_model.predict(util.normalize(X_train)) # sized at (m,7,7,1024)
	A_transfer = np.reshape(A_transfer,(m,-1)) # sized at (m,7*7*1024)

	# Train our model on output of Resnet50
	our_model = util.neural_network(A_transfer, y_train,[1024,256,64,16,6],epochs = 15,lambd=1e-4)
	print(our_model.summary())

	# Prediction on training data
	y_train_pred = our_model.predict(A_transfer)

	np.savetxt("confusion_matrix_training_DenseNet.csv", confusion_matrix.generate_confusion_matrix(y_train_pred, y_train), delimiter=',')

	#------ Validation ------
	# Run validation data through RestNet50
	X_valid, y_valid = data_proc.get_batch('archive/one-indexed-files-notrash_val.txt',224,224)
	m = X_valid.shape[0]
	A_transfer = transfer_model.predict(util.normalize(X_valid)) # sized at (m,7,7,2048)
	A_transfer = np.reshape(A_transfer,(m,-1)) # sized at (m,7*7*2048)

	# Prediction on validation data
	y_valid_pred = our_model.predict(A_transfer)

	np.savetxt("confusion_matrix_valid_DenseNet.csv", confusion_matrix.generate_confusion_matrix(y_valid_pred, y_valid), delimiter=',')



	# -------- Printing metrics for Training and Validation -------
	acc = util.compute_accuracy(y_train_pred, y_train)
	prec = util.compute_ma_precision(y_train_pred, y_train)
	recall = util.compute_ma_recall(y_train_pred, y_train)
	F1 = util.compute_ma_F1Score(y_train_pred, y_train)

	print("Metrics for Training - DenseNet121:")
	print(f"Accuracy = {acc}")
	print(f"Precision = {prec}")
	print(f"Recall = {recall}")
	print(f"F1 score = {F1}")

	acc = util.compute_accuracy(y_valid_pred, y_valid)
	prec = util.compute_ma_precision(y_valid_pred, y_valid)
	recall = util.compute_ma_recall(y_valid_pred, y_valid)
	F1 = util.compute_ma_F1Score(y_valid_pred, y_valid)

	print("Metrics for Validation - DenseNet121:")
	print(f"Accuracy = {acc}")
	print(f"Precision = {prec}")
	print(f"Recall = {recall}")
	print(f"F1 score = {F1}")



if __name__ == "__main__":
    main()
