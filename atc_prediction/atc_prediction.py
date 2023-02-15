import pickle
from tqdm import tqdm
from collections import Counter

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import xgboost as xgb



def get_pred_candidates_from_pred_prob(predict_proba: list) -> list:
    '''
    get predicted ATC class from probabilities predicted by XBGoost
    '''
    predict_proba_desc = sorted(predict_proba, reverse=True)

    pred_cands = []
    for pp in predict_proba_desc:
        pred_cands.append(predict_proba.index(pp))
        
    return pred_cands



#setting for atc prediction
pred_settings = {'MCTN_PROP': 90, 'DROP_PK': False}
#import dataset for atc prediciton
with open(f'used_X_mctn90_atc14_PK_False.p', 'rb') as f:
    X = pickle.load(f)
with open(f'used_y_mctn90_atc14_PK_False.p', 'rb') as f:
    y = pickle.load(f)  
    
#dataset statistics
print(f'Number of samples: {len(y)}')
print(f'ATC level distribution - major levels: {Counter(y).most_common()}')
print(f'Dimension of X: {len(X[0])}')

#split dataset (train/test)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=42,
                                                    shuffle=True)
#tranfrom y to binary format e.g., (0,1,1 ..) 
y_train_class = list(set(y_train)); y_train_class.sort()
y_train = [y_train_class.index(y_ind) for y_ind in y_train]
y_test = [y_train_class.index(y_ind) for y_ind in y_test]

#model training
xgb.set_config(verbosity=0)
    
#Gridsearch to find the best hyperparmaeter setting
#Setting Parameters (grid)
param_grid = {'learning_rate': [0.1, 0.05, 0.01],
                'max_depth': [2, 6, 8],
                'n_estimator': [10, 100, 1000]}
#model set
xgb_model_grid = xgb.XGBClassifier(objective='multi:softprob', num_classes=14)
clf = GridSearchCV(xgb_model_grid, param_grid, verbose=1)
clf.fit(X_train, y_train)
#best model
xgb_model_grid = clf
best_params = xgb_model_grid.best_params_
    
#training and evaluation (without grid)
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_classes=14, **best_params)
xgb_model.fit(X_train, y_train) 
y_pred_int = xgb_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix  
print(best_params)
print(confusion_matrix(y_test, y_pred_int))
print(classification_report(y_test, y_pred_int))

#leave-on-out evaluation
y_test_total = []; y_pred14_total = []
for sample_idx in tqdm(range(len(X))):
    #split train/test data in loo evaluation setting
    X_train, X_test, y_train, y_test = (X[:sample_idx]+X[sample_idx+1:],
                                        X[sample_idx],
                                        y[:sample_idx]+y[sample_idx+1:],
                                        y[sample_idx])

    #tranfrom y to binary format e.g., (0,1,1 ..) 
    y_train_class = list(set(y_train)); y_train_class.sort()
    y_train = [y_train_class.index(y_ind) for y_ind in y_train]
    y_test = [y_train_class.index(y_ind) for y_ind in y_test]

    #XGBoost model training
    xgb.set_config(verbosity=0)
    xgb_model = xgb.XGBClassifier(objective='multi:softprob', 
                                  num_classes=14, 
                                  **best_params)
    xgb_model.fit(X_train, y_train)
    
    #Evaluation - leave-on-out
    y_pred14 = get_pred_candidates_from_pred_prob(list(xgb_model.predict_proba([X_test])[0]))
    y_test_total.append(y_test[0])
    y_pred14_total.append(y_pred14)

#calculate loo accuracy for 14 ATC classes
accuracies = {}
for pred_order in range(14):
    y_pred_total = [y_pred14[pred_order] for y_pred14 in y_pred14_total]
    pred_order_cor = [y_pred for y_test, y_pred in zip(y_test_total, y_pred_total) 
                        if y_test==y_pred]
    
    accuracies[pred_order] = len(pred_order_cor)/len(y_test_total)