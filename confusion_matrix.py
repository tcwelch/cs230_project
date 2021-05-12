import numpy as np
import pandas as pd

# file to visualize confusion matrix (to be exported to Excel)
# editing again for test


pred_test = np.asarray([[0, 0, 1], [0, 0, 1], [0, 1, 0]])
label_test = np.asarray([[1, 0, 0], [0, 0, 1], [0, 0, 1]])

def main():
    np.savetxt("test2.csv", generate_confusion_matrix(pred_test, label_test), delimiter=',')

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
        print("row")
        print(row)
        col = np.argmax(preds[i,:])
        print("col")
        print(col)
        res[row,col] += 1
    return res.astype(int)

# Compute overall prediction accuracy
def compute_accuracy(preds, labels):
    predict_labels = format_predictions(preds)
    accuracy_vec = predict_labels == labels
    overall_accuracy = mean(accuracy_vec)
    return overall_accuracy

# Compute micro-average precision
def compute_ma_precision(preds, labels):
    predict_labels = format_predictions(preds)
    # Get list of all true positives
    tp_total = predict_labels[predict_labels == labels]
    tp_arr = np.zeros((6,1))
    fp_arr = np.zeros((6,1))
    tp_arr[0] = len([x for x in tp_total if argmax(x) == 0])
    fp_arr[0] = len([x for x in predict_labels if argmax(x) == 0]) - tp_arr[0]
    tp_arr[1] = len([x for x in tp_total if argmax(x) == 1])
    fp_arr[1] = len([x for x in predict_labels if argmax(x) == 1]) - tp_arr[1]
    tp_arr[2] = len([x for x in tp_total if argmax(x) == 2])
    fp_arr[2] = len([x for x in predict_labels if argmax(x) == 2]) - tp_arr[2]
    tp_arr[3] = len([x for x in tp_total if argmax(x) == 3])
    fp_arr[3] = len([x for x in predict_labels if argmax(x) == 3]) - tp_arr[3]
    tp_arr[4] = len([x for x in tp_total if argmax(x) == 4])
    fp_arr[4] = len([x for x in predict_labels if argmax(x) == 4]) - tp_arr[4]
    tp_arr[5] = len([x for x in tp_total if argmax(x) == 5])
    fp_arr[5] = len([x for x in predict_labels if argmax(x) == 5]) - tp_arr[5]
    ma_precision = np.sum(tp_arr) / (np.sum(tp_arr) + np.sum(fp_arr))
    return ma_precision

# Compute precision
def compute_ma_precision(preds, labels):
    predict_labels = format_predictions(preds)
    # Get list of all true positives
    tp_total = predict_labels[predict_labels == labels]
    tp_arr = np.zeros((6,1))
    fn_arr = np.zeros((6,1))
    tp_arr[0] = len([x for x in tp_total if argmax(x) == 0])
    tp_arr[1] = len([x for x in tp_total if argmax(x) == 1])
    tp_arr[2] = len([x for x in tp_total if argmax(x) == 2])
    tp_arr[3] = len([x for x in tp_total if argmax(x) == 3])
    tp_arr[4] = len([x for x in tp_total if argmax(x) == 4])
    tp_arr[5] = len([x for x in tp_total if argmax(x) == 5])

    fn_arr[0] = len([x for x in predict_labels if argmax(x) != 0]) - (np.sum(tp_arr) - tp_arr[0])
    fn_arr[1] = len([x for x in predict_labels if argmax(x) != 1]) - (np.sum(tp_arr) - tp_arr[1])
    fn_arr[2] = len([x for x in predict_labels if argmax(x) != 2]) - (np.sum(tp_arr) - tp_arr[2])
    fn_arr[3] = len([x for x in predict_labels if argmax(x) != 3]) - (np.sum(tp_arr) - tp_arr[3])
    fn_arr[4] = len([x for x in predict_labels if argmax(x) != 4]) - (np.sum(tp_arr) - tp_arr[4])
    fn_arr[5] = len([x for x in predict_labels if argmax(x) != 5]) - (np.sum(tp_arr) - tp_arr[5])

    ma_recall = np.sum(tp_arr) / (np.sum(tp_arr) + np.sum(fn_arr))
    return ma_recall

# Compute F1

if __name__ == "__main__":
    main()