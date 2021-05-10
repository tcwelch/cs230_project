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



if __name__ == "__main__":
    main()