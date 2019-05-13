#
# @Author: Songyang Zhang 
# @Date: 2018-11-16 22:57:29 
# @Last Modified by:   Songyang Zhang 
# @Last Modified time: 2018-11-16 22:57:29 
#
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def accuracy(preds, targets, threshold=0.0):
    """
    Args:
        preds:

        targets:
    """
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    preds = (preds > threshold).astype(np.float32)
    targets = targets.astype(np.float32)
    mean_acc = (preds==targets).astype(np.float32).mean()
    return mean_acc

def calc_acc_over_epoch(preds, targets, threshold=0.5):

    preds = (preds > threshold).astype(np.float32)
    targets = targets.astype(np.float32)
    acc_each_cls = (preds == targets).astype(np.float32).mean(0)
    return acc_each_cls


def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))
    

# def F1_soft(preds, targests, threshold=0.5, d=50.0):
#     ""
#     preds = sigmoid_np(d*(preds - threshold))
#     targests = targests.astype(np.float())
#     score = 2.0*(preds*targests).sum(axis=0) / ((preds+targests).sum(axis=0) + 1e-6)
#
#     return score

def F1_score(preds, targets, threshold=0.5):

    preds = (preds>threshold).astype(np.int32)
    targets = targets.astype(np.int32)
    macro_f1 = f1_score(y_true=targets, y_pred=preds, average='macro')
    micro_f1 = f1_score(y_true=targets, y_pred=preds, average='micro')

    return macro_f1, micro_f1

def calc_roc_auc(preds, targets):

    preds = preds.astype(np.float32)
    targets = targets.astype(np.float32)

    auc = roc_auc_score(y_true=targets, y_score=preds, average=None)
    mean_auc = np.array(auc).mean()

    return auc, mean_auc
