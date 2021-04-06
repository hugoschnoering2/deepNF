
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

def real_AUPR(label, score):
    label = label.flatten()
    score = score.flatten()
    order = np.argsort(score)[::-1]
    label = label[order]
    P = np.count_nonzero(label)
    TP = np.cumsum(label, dtype=float)
    PP = np.arange(1, len(label)+1, dtype=float)
    x = np.divide(TP, P)
    y = np.divide(TP, PP)
    pr = np.trapz(y, x)
    f = np.divide(2*x*y, (x + y))
    idx = np.where((x + y) != 0)[0]
    if len(idx) != 0:
        f = np.max(f[idx])
    else:
        f = 0.0
    return pr, f

def evaluate_performance(y_test, y_score, y_pred):
    n_classes = y_test.shape[1]
    perf = dict()
    perf["M-aupr"] = 0.0
    n = 0
    for i in range(n_classes):
        perf[i], _ = real_AUPR(y_test[:, i], y_score[:, i])
        if sum(y_test[:, i]) > 0:
            n += 1
            perf["M-aupr"] += perf[i]
    perf["M-aupr"] /= n
    perf["m-aupr"], _ = real_AUPR(y_test, y_score)
    perf['acc'] = accuracy_score(y_test, y_pred)
    alpha = 3
    y_new_pred = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha*[1])
    perf["F1"] = f1_score(y_test, y_new_pred, average='micro')
    return perf
