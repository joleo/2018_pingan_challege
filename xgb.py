# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     parner_xgb
   Description :
   Author :       liyang
   date：          2018/4/19 0019
-------------------------------------------------
   Change Activity:
                   2018/4/19 0019:
-------------------------------------------------
"""
__author__ = 'liyang'
from sklearn import metrics
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import fbeta_score


if  __name__ == '__main__':
    train_df = pd.read_csv('data/train_down_data.csv')
    test_df = pd.read_csv('data/test_modifier.csv')
    target = 'acc_now_delinq'
    IDcol = 'member_id'

    xgb = XGBClassifier(
        learning_rate=0.1,
        n_estimators=250,
        max_depth=5,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=2,
        eta=0.025,
        gamma=0,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    isCV = True;
    cv_folds = 3;
    early_stopping_rounds = 10
    predictors = [x for x in train_df.columns if x not in [target, IDcol]]


    if isCV:
        xgb_param = xgb.get_xgb_params()
        xgtrain = xgb.DMatrix(train_df[predictors].values, label=train_df[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        xgb.set_params(n_estimators=cvresult.shape[0])

    xgb.fit(train_df[predictors], train_df['acc_now_delinq'], eval_metric='auc')
    dtrain_predictions = xgb.predict(train_df[predictors])
    dtrain_predprob = xgb.predict_proba(train_df[predictors])[:, 1]

    # 打印结果
    print("Accuracy : %.4g" % metrics.accuracy_score(train_df['acc_now_delinq'].values, dtrain_predictions))
    print("Recall:%.4g" % metrics.recall_score(train_df['acc_now_delinq'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_df['acc_now_delinq'], dtrain_predprob))
    print("f2_score:%.4g" % fbeta_score(train_df['acc_now_delinq'].values, dtrain_predictions, beta=2))

    #  提交文件
    dtest_predictions = xgb.predict(test_df[predictors])
    test_prediction = np.array(dtest_predictions)
    test_df.loc[:, 'acc_now_delinq'] = test_prediction
    submission = test_df.loc[:, ['member_id', 'acc_now_delinq']]
    submission.to_csv('data/summission.csv', index=False)
