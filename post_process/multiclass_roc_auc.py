import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp


def multiclass_roc_auc_score(y_true, y_score):
'''
y_true: ground truth label, should be a numpy array with shape [n_samples, n_classes]
        the label should be one hot label
y_score: predicted score, should be a numpy array with shape [n_samples, n_classes]
         the score could be probability, measure of decisions or confidence value 
'''
    assert y_true.shape == y_score.shape
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_true.shape[1]

    '''compute ROC curve and ROC area for each class'''
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    '''compute micro-average ROC curve and ROC area'''
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    '''compute macro-average ROC curve and ROC area'''
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

    return roc_auc
