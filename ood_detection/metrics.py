import numpy as np
from sklearn import metrics


def compute_all_metrics(conf, mask):
    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, mask, recall)

    return {"fpr": fpr, "auroc": auroc, "aupr_in": aupr_in, "aupr_out": aupr_out}


# auc
def auc_and_fpr_recall(conf, mask, tpr_th):
    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(mask, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(mask, -conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(1 - mask, conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr
