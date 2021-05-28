import data_proc 
import util
import numpy as np
import tensorflow as tf
import confusion_matrix


def main():
	#------ Training ------
	print("----------Getting data----------")
	X1, y1 = data_proc.get_batch('archive/one-indexed-files-notrash_train.txt',224,224)
	print("----------Adding augmented data----------")
	aug_X1,aug_y1 = data_proc.get_aug(X1,y1,[646, 597, 0, 653, 714, 0])#[146, 97, 0, 153, 214, 0]) #[146, 97, 213, 153, 214, 409])
	X1.extend(aug_X1)
	y1.extend(aug_y1)
	X_train = np.array(X1)
	y_train = np.array(y1)
	m = X_train.shape[0]

	# Run training data through RestNet50
	print("----------Processing data in transfer network (Training data) ----------")
	transfer_model = tf.keras.applications.DenseNet121(include_top=False,weights="imagenet")
	A_transfer = transfer_model.predict(util.normalize(X_train)) # sized at (m,7,7,1024)
	A_transfer = np.reshape(util.normalize(A_transfer),(m,-1)) # sized at (m,7*7*1024)

	# Train our model on output of Resnet50
	print("----------Training our DLNN----------")
	our_model = util.neural_network(A_transfer, y_train,[1024,256,64,16,6],epochs = 5,lambd=1e-4)
	print(our_model.summary())

	# Prediction on training data
	print("----------Predicting on training set----------")
	y_train_pred = our_model.predict(A_transfer)

	np.savetxt("confusion_matrix_training_DenseNet.csv", confusion_matrix.generate_confusion_matrix(y_train_pred, y_train), delimiter=',')

	#------ Validation ------
	# Run validation data through RestNet50
	print("----------Processing data in transfer network (Validation data) ----------")
	X2, y2 = data_proc.get_batch('archive/one-indexed-files-notrash_val.txt',224,224)
	X_valid = np.array(X2)
	y_valid = np.array(y2)
	m = X_valid.shape[0]
	A_transfer = transfer_model.predict(util.normalize(X_valid)) # sized at (m,7,7,2048)
	A_transfer = np.reshape(util.normalize(A_transfer),(m,-1)) # sized at (m,7*7*2048)

	# Prediction on validation data
	print("----------Predicting on training set----------")
	y_valid_pred = our_model.predict(A_transfer)

	np.savetxt("confusion_matrix_valid_DenseNet.csv", confusion_matrix.generate_confusion_matrix(y_valid_pred, y_valid), delimiter=',')



	# -------- Printing metrics for Training and Validation -------
	print("----------Computing Metrics----------")
	acc = confusion_matrix.compute_accuracy(y_train_pred, y_train)
	recall, prec = confusion_matrix.compute_ma_precision_recall(y_train_pred, y_train)
	# recall = confusion_matrix.compute_ma_recall(y_train_pred, y_train)
	F1 = confusion_matrix.compute_ma_F1Score(y_train_pred, y_train)

	print("Metrics for Training - DenseNet121:")
	print(f"Accuracy = {acc}")
	print(f"Precision = {prec}")
	print(f"Recall = {recall}")
	print(f"F1 score = {F1}")

	acc = confusion_matrix.compute_accuracy(y_valid_pred, y_valid)
	recall, prec = confusion_matrix.compute_ma_precision_recall(y_valid_pred, y_valid)
	# recall = confusion_matrix.compute_ma_recall(y_valid_pred, y_valid)
	F1 = confusion_matrix.compute_ma_F1Score(y_valid_pred, y_valid)

	print("Metrics for Validation - DenseNet121:")
	print(f"Accuracy = {acc}")
	print(f"Precision = {prec}")
	print(f"Recall = {recall}")
	print(f"F1 score = {F1}")

	# confusion_matrix.plot_ROC(y_train_pred, y_train,'DenseNet_ROC_train.jpeg')
	confusion_matrix.plot_ROC(y_valid_pred, y_valid,'DenseNet_ROC_valid.jpeg')



if __name__ == "__main__":
    main()
