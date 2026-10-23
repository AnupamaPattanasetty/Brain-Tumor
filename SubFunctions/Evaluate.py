from sklearn.metrics import matthews_corrcoef, mean_squared_error
from mealpy.metrics import confusion_matrix

import math



def tt(value):
    power = math.ceil(math.log10(value) - 1)
    A1 = 100 ** (math.log10(value) - power)
    return A1


def Evaluation_Metrics(y, y_pred):
    """
    :param y_pred: Model predicted labels
    :param y: Original labels
    :return: return values is the evaluation metrics

    we get confusion matrix from the predictions
    Then implement the formula for getting the evaluation metrics from the confusion matrix
    """

    cnf_matrix, y_true, y_pred = confusion_matrix(y.astype(int), y_pred.astype(int))
    # cnf_matrix = np.sum(cnf_matrix, axis=0)

    TP = cnf_matrix[0, 0]
    FN = cnf_matrix[0, 1]
    FP = cnf_matrix[1, 0]
    TN = cnf_matrix[1, 1]



    SEN = TP / (TP + FN)  # Sensitivity
    SPE = TN / (TN + FP)  # Specificity
    ACC = (TP + TN) / (TP + TN + FP + FN)
    FMS = (2 * TP) / (2 * TP + FP + FN)
    PRE = TP / (TP + FP)
    REC = SEN
    BER = tt(mean_squared_error(y_true, y_pred))
    MCC = matthews_corrcoef(y_true, y_pred)
    TS = TP / (TP + FP + FN)  # Threat score
    NPV = TN / (TN + FN)  # negative predictive value
    PPV = TP / (TP + FP)
    FOR = FN / (FN + TN)  # false omission rate
    F1 = 2 * TP / ((2 * TP) + FP + FN)
    # print([[TP, TN], [FP, FN]])     #

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    return [ACC, SEN, SPE, PRE, F1]

def Evaluation_Metrics1(y, y_pred):
    """
    :param y_pred: Model predicted labels
    :param y: Original labels
    :return: return values is the evaluation metrics

    we get confusion matrix from the predictions
    Then implement the formula for getting the evaluation metrics from the confusion matrix
    """

    cnf_matrix, y_true, y_pred = confusion_matrix(y.astype(int), y_pred.astype(int))
    # cnf_matrix = np.sum(cnf_matrix, axis=0)

    TP = cnf_matrix[0, 0]
    FN = cnf_matrix[0, 1]
    FP = cnf_matrix[1, 0]
    TN = cnf_matrix[1, 1]

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return [TPR, FPR]