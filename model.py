#imports
import pandas as pd
from sklearn.model_selection import train_test_split
from eval_metric_code import get_eval_metric
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm
import random

# =============================================================================
# In this file we predict the exact delay then convert it to delay bins using GBT regressor
# and a thresholding method to find the optimal bin for every range of predictions
# =============================================================================

def thresholding(y_pred, thresholds):
    for i in range(len(y_pred)):
        for j in range(len(thresholds)):
            if y_pred[i] < thresholds[j]:
                y_pred[i] = j
                break
            elif j == len(thresholds)-1:
                y_pred[i] = j+1
    return y_pred

def determine_thresholds(y_true, y_pred, n_iter=300):
    """We need to determine optimum threshold for each class"""
    best_thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    best_result = [16]
    for i in tqdm(range(n_iter)):
        if i%20 == 0: 
            print(best_result[-1], end=" ")
        thresholds = [random.random()]
        for i in range(1,7):
            thresholds.append(random.uniform(thresholds[i-1], i+1))
        _y_pred = thresholding(y_pred.copy(), thresholds)
        
        result = get_eval_metric(y_true, _y_pred)
        if result < best_result[-1]:
            best_result.append(result)
            best_thresholds = thresholds
    return best_result, best_thresholds

# dataset created by GBT_preprocess
data = pd.read_csv('data/processed_train.csv')
X = data.drop(['delay_bin', 'exact_delay', 'arrival_datetime_act'], axis = 1)
# exact delay in minutes divided by 20 minutes
y = data['exact_delay']/20

# For Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=41)

# =============================================================================
# GBT Regressor

model = GradientBoostingRegressor(max_depth = 12)
model.fit(X_train, y_train)
#predict exact delay
y_pred_val = model.predict(X_val)
#true delays for validation set
y_true_val = data.loc[y_val.index]['delay_bin'].to_list()
# find the best thresholds on validaton set
res, thresholds = determine_thresholds(y_true_val, y_pred_val, n_iter=10)

#compute error on test set
y_pred_test = model.predict(X_test)
#true delays for validation set
y_true_test = data.loc[y_test.index]['delay_bin'].to_list()
# change predictions using thresholding of the validation set
y_pred_test = thresholding(y_pred_test, thresholds)

print("the score on the test set is {}".format(get_eval_metric(y_true_test, y_pred_test)))
