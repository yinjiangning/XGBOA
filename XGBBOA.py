# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 09:57:24 2021

@author: Administrator
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

data = pd.read_csv('D:/s-sn/ASCII1/evidence1.csv',encoding='gbk')



x = data.drop('status',axis=1)
y = data.status
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state =2018, shuffle = True)


'''
xg = xgb.XGBClassifier()
#不做优化的结果
print(np.mean(cross_val_score(xg,X_train,y_train,scoring="accuracy",cv=20)))
'''

def xg_cv(learning_rate,
        max_depth,
        gamma,
        min_child_weight,
        max_delta_step,
        subsample,
        colsample_bytree):
    
      
    val = cross_val_score(XGBClassifier(learning_rate=learning_rate,
        max_depth=int(max_depth),
        gamma=gamma,
        min_child_weight=min_child_weight,
        max_delta_step=max_delta_step,
        subsample=subsample,
        colsample_bytree=colsample_bytree),
            X_train,y_train,scoring="accuracy",cv=5).mean()
    return val
#贝叶斯优化
xg_bo = BayesianOptimization(xg_cv,
                             {
    'learning_rate': (0.01, 1),
    'max_depth': (2, 12),
    'gamma': (0.001, 10.0),
    'min_child_weight': (0, 20),
    'max_delta_step': (0, 10),
    'subsample': (0.4, 1.0),
    'colsample_bytree': (0.4, 1.0)
                             })
#开始优化
num_iter = 25
init_points = 5
xg_bo.maximize(init_points=init_points,n_iter=num_iter)
#显示优化结果
xg_bo.res["max"]
#附近搜索（已经有不错的参数值的时候）

'''
xg_bo.explore(
    {'learning_rate': [0.01, 0.03, 0.01, 0.03, 0.1, 0.3, 0.1, 0.3],
    'max_depth': [3, 8, 3, 8, 8, 3, 8, 3],
    'gamma': [0.5, 8, 0.2, 9, 0.5, 8, 0.2, 9],
    'min_child_weight': [0.2, 0.2, 0.2, 0.2, 12, 12, 12, 12],
    'max_delta_step': [1, 2, 2, 1, 2, 1, 1, 2],
    'subsample': [0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],
    'colsample_bytree': [0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8]
    })

#验证优化后参数的结果
rf =xgb(max_depth=5, max_features=0.432, min_samples_split=2, n_estimators=190)
np.mean(cross_val_score(rf, X_train, y_train, cv=20, scoring='roc_auc'))
'''

history_df = pd.DataFrame(xg_bo.res['all']['params'])
history_df2 = pd.DataFrame(xg_bo.res['all']['values'])
history_df = pd.concat((history_df, history_df2), axis=1)
history_df.rename(columns={0: 'gini'}, inplace=True)
history_df['AUC'] = (history_df['gini'] + 1) / 2
history_df.to_csv('D:/Pddd.csv')
