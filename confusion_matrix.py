import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

def main():
    pred_test = np.asarray([[0, 0, 1,0, 0, 0], [0, 0, 1,0, 0, 0], [1, 0, 0,0, 0, 0]])
    label_test = np.asarray([[1, 0, 0,0, 0, 0], [0, 0, 1,0, 0, 0], [1, 0, 0,0, 0, 0]])
    np.savetxt("test2.csv", generate_confusion_matrix(pred_test, label_test), delimiter=',')
    print("precision test: " + str(compute_ma_precision_recall(pred_test, label_test)))
    plot_ROC(pred_test, label_test,'test_img.jpeg')

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


# Compute micro averaged precision. USE THIS METHOD
def compute_ma_precision_recall(preds, labels):
    prediction_labels = format_predictions(preds)

    # Compute true positives
    tp = prediction_labels * labels
    # Compute false positives
    fp = prediction_labels > labels
    fn = prediction_labels < labels
    num_tp = np.sum(tp)
    num_fp = np.sum(fp)
    num_fn = np.sum(fn)

    ma_prec = num_tp / (num_tp + num_fp)
    ma_recall = num_tp / (num_tp + num_fn)
    return (ma_recall, ma_prec)


            

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
    ma_recall, ma_precision = compute_ma_precision_recall(preds, labels)
    return 2/((1/ma_recall) + (1/ma_precision))

# plot ROC curves for multilabel
def plot_ROC(preds, labels, save_path):
    predict_labels = format_predictions(preds)
    n_classes = labels.shape[1]
    
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(0,n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predict_labels[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predict_labels.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['purple','magenta','aqua','cornflowerblue','lime','darkorange'])
    classes = ['glass','paper','cardboard','plastic','metal','trash']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    main()