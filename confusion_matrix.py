import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd


def main():
    pred_test = np.asarray([[0, 0, 1], [0, 0, 1], [1, 0, 0]])
    label_test = np.asarray([[0, 0, 1], [0, 0, 1], [1, 0, 0]])
    np.savetxt("test2.csv", generate_confusion_matrix(pred_test, label_test), delimiter=',')
    print("recall test: " + str(compute_ma_recall(pred_test, label_test)))

# convert softmax output to binary values
def format_predictions(softmax_output):
    res = np.zeros(softmax_output.shape)
    for i in range(softmax_output.shape[0]):
        res[i,:] = softmax_output[i,:] == np.max(softmax_output[i,:])
    return res

# return confusion matrix
def generate_confusion_matrix(preds, labels):
    num_ex = labels.shape[0]
    predicted_labels = format_predictions(preds)
    width = labels.shape[1]
    height = width
    res = np.zeros((width, height))
    for i in range(num_ex):
        row = np.argmax(labels[i,:])
        col = np.argmax(preds[i,:])
        res[row,col] += 1
    return res.astype(int)

# Compute overall prediction accuracy
def compute_accuracy(preds, labels):
    predict_labels = format_predictions(preds)
    accuracy_vec = predict_labels == labels
    overall_accuracy = np.mean(accuracy_vec)
    return overall_accuracy

# Compute micro-average precision 
def compute_ma_precision(preds, labels):
    predict_labels = format_predictions(preds)
    tp_arr = np.zeros(6)
    fp_arr = np.zeros(6)
    tp_arr[0] = len([x for x in labels if np.argmax(x) == 0])
    fp_arr[0] = max(len([x for x in predict_labels if np.argmax(x) == 0]) - tp_arr[0], 0)
    print(fp_arr[0])
    tp_arr[1] = len([x for x in labels if np.argmax(x) == 1])
    fp_arr[1] = max(len([x for x in predict_labels if np.argmax(x) == 1]) - tp_arr[1],0)
    print(fp_arr[1])
    tp_arr[2] = len([x for x in labels if np.argmax(x) == 2])
    fp_arr[2] = max(len([x for x in predict_labels if np.argmax(x) == 2]) - tp_arr[2],0)
    print(fp_arr[2])
    tp_arr[3] = len([x for x in labels if np.argmax(x) == 3])
    fp_arr[3] = max(len([x for x in predict_labels if np.argmax(x) == 3]) - tp_arr[3],0)
    print(fp_arr[3])
    tp_arr[4] = len([x for x in labels if np.argmax(x) == 4])
    fp_arr[4] = max(len([x for x in predict_labels if np.argmax(x) == 4]) - tp_arr[4],0)
    print(fp_arr[4])
    tp_arr[5] = len([x for x in labels if np.argmax(x) == 5])
    fp_arr[5] = max(len([x for x in predict_labels if np.argmax(x) == 5]) - tp_arr[5],0)
    print(fp_arr[5])
    ma_precision = np.sum(tp_arr) / (np.sum(tp_arr) + np.sum(fp_arr))
    return ma_precision

# Compute recall
def compute_ma_recall(preds, labels):
    predict_labels = format_predictions(preds)
    tp_arr = np.zeros(6)
    fn_arr = np.zeros(6)
    tp_arr[0] = len([x for x in labels if np.argmax(x) == 0])
    tp_arr[1] = len([x for x in labels if np.argmax(x) == 1])
    tp_arr[2] = len([x for x in labels if np.argmax(x) == 2])
    tp_arr[3] = len([x for x in labels if np.argmax(x) == 3])
    tp_arr[4] = len([x for x in labels if np.argmax(x) == 4])
    tp_arr[5] = len([x for x in labels if np.argmax(x) == 5])

    fn_arr[0] = max(len([x for x in predict_labels if np.argmax(x) != 0]) - (np.sum(tp_arr) - tp_arr[0]),0)
    fn_arr[1] = max(len([x for x in predict_labels if np.argmax(x) != 1]) - (np.sum(tp_arr) - tp_arr[1]),0)
    fn_arr[2] = max(len([x for x in predict_labels if np.argmax(x) != 2]) - (np.sum(tp_arr) - tp_arr[2]),0)
    fn_arr[3] = max(len([x for x in predict_labels if np.argmax(x) != 3]) - (np.sum(tp_arr) - tp_arr[3]),0)
    fn_arr[4] = max(len([x for x in predict_labels if np.argmax(x) != 4]) - (np.sum(tp_arr) - tp_arr[4]),0)
    fn_arr[5] = max(len([x for x in predict_labels if np.argmax(x) != 5]) - (np.sum(tp_arr) - tp_arr[5]),0)

    ma_recall = np.sum(tp_arr) / (np.sum(tp_arr) + np.sum(fn_arr))
    return ma_recall

# Compute F1
def compute_ma_F1Score(preds, labels):
    ma_recall = compute_ma_recall(preds, labels)
    ma_precision = compute_ma_precision(preds, labels)
    return 2/((1/ma_recall) + (1/ma_precision))

if __name__ == "__main__":
    main()