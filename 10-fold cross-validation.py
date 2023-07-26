#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:09:30 2023
@author: Rui Cheng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
##############################################################################
internal = pd.read_csv('internal.txt', sep='\t')
y_internal = np.array(internal['target'])
X_external = np.array(internal.drop(['target'], 1))
external = pd.read_csv('external.txt', sep='\t')
y_external = np.array(external['target'])
X_external = np.array(external.drop(['target'], 1))
##############################################################################
cv = KFold(n_splits=10,shuffle=True)
#1.Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=2000,random_state=90,max_depth=20,min_samples_leaf=10
                                   ,min_samples_split=2,max_features=5,criterion='gini',
                                   oob_score=True,max_leaf_nodes=None,min_impurity_decrease=0)

#2.xgboost
# classifier = XGBClassifier(n_estimators=110,max_depth=4,min_child_weight=1,gamma=0.1
#                    ,subsample=0.5,colsample_bytree=0.7,eta=0.1,learning_rate= 0.1)

#3.lightgbm
#classifier = LGBMClassifier(boosting_type='gbdt',objective='binary',learning_rate=0.5, n_estimators=95, max_depth=4, num_leaves=5,max_bin=7,
#                                min_data_in_leaf=5,bagging_fraction=0.7,bagging_freq= 15, feature_fraction= 0.6,lambda_l1=0.001,
#                                lambda_l2=0.3,min_split_gain=0)

#4.SVM
#classifier = SVC(C=2,kernel='rbf',gamma=0.1)

#5.gcForst
#classifier = gcForest(shape_1X=38, window=11, tolerance=0.6,min_samples_mgs=10, min_samples_cascade=7)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(1, sharey=True, figsize=(14, 10.5))

for i, (train, test) in enumerate(cv.split(X,y)):
    classifier.fit(X.iloc[train], y.iloc[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X.iloc[test],
        y.iloc[test],
        name="ROC fold {}".format(i+1),
        alpha=0.3,
        lw=1,
        ax=ax
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean derivation cohort (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
)
ax.legend(loc="lower right")
##############################################################################
y_external_probs = classifier.predict_proba(X_external)[:, 1]
fpr_external, tpr_external, _ = roc_curve(y_external, y_external_probs)
roc_auc_external = auc(fpr_external, tpr_external)
##############################################################################
plt.plot(fpr_external, tpr_external, color='green', lw=2, label='validation cohort (AUC = %0.2f)' % roc_auc_external)
plt.savefig('rf.pdf', dpi=300, bbox_inches="tight")
plt.show()
