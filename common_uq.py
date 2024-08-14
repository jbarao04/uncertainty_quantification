import numpy as np
from collections import namedtuple
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def jackknife_predict(modelo,X_train,y_train,X_test,config):
    d_value=config['d-value']
    alpha=config['alpha']
    ij_samples=len(X_train)//d_value
    y = np.zeros([ij_samples, len(X_test)])
    for i in np.arange(ij_samples):
        start_idx = i * d_value
        end_idx = start_idx + d_value
        X_removed = X_train[start_idx:end_idx]
        y_removed = y_train[start_idx:end_idx]
        X_train_new = np.delete(X_train, np.s_[start_idx:end_idx], axis=0)
        y_train_new = np.delete(y_train, np.s_[start_idx:end_idx], axis=0)
        modelo.fit(X_train_new,y_train_new)
        y[i] = modelo.predict_proba(X_test)[:,1]
        X_train = np.insert(X_train_new, start_idx, X_removed, axis=0)
        y_train = np.insert(y_train_new, start_idx, y_removed, axis=0)
    y_lower = np.quantile(y, q=0.5 * alpha, axis=0)
    y_upper = np.quantile(y, q=(1. - 0.5 * alpha), axis=0)
    y_pred=np.mean(y,axis=0)
    Result = namedtuple('res', ['y','y_pred', 'y_lower', 'y_upper'])
    res = Result(y,y_pred, y_lower, y_upper)

    return res

def plot_uq_quality(test_targets,y_pred_xgb,y_ub,y_lb,bins_num):
    y_true = test_targets
    prob_true, prob_pred = calibration_curve(y_true, y_pred_xgb, n_bins=bins_num)

    prob_true_ub, prob_pred_ub = calibration_curve(y_true, y_ub, n_bins=bins_num)

    prob_true_lb, prob_pred_lb = calibration_curve(y_true, y_lb, n_bins=bins_num)

    plt.plot(prob_pred, prob_true, marker='o', label='Mean predicted')
    plt.fill_between(prob_pred, prob_true_lb, prob_true_ub, color='lightblue', alpha=0.5, label='Uncertainty Region')
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.legend()
    plt.tight_layout()
    plt.show()