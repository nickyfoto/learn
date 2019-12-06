"""provide some helper function for sklearn"""

import numpy as np
from scipy.optimize import linprog
from sklearn.preprocessing import StandardScaler
# for row_index, (input, prediction, label) in enumerate(zip (X_test, svm_predictions, y_test)):
#   if prediction != label:
#     print('Row', row_index, 'has been classified as ', prediction, 'and should be ', label)

def lp(unique_y, X, y):
    ls = []
    for i in unique_y:
        t = np.where(y == i, 1 , -1).reshape(-1, 1)
        A_ub = np.append(X * t, t, 1)
        b_ub = np.repeat(-1, A_ub.shape[0]).reshape(-1,1)
        c_obj = np.repeat(1, A_ub.shape[1])
        res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub,
                  options={"disp": False})
        # print(res.success)
        ls.append(res.success)
    return ls

def linear_separability(X, y, scale=True):
    """
    Given X, y of a classfication problem, provide the boolean 
    linear separability among classes.
    http://www.tarekatwan.com/index.php/2017/12/methods-for-testing-linear-separability-in-python/
    """
    unique_y = np.unique(y)
    if scale:
        X = StandardScaler().fit_transform(X)
    ls = lp(unique_y, X, y)
    return {k: v for k, v in zip(unique_y, ls)}
