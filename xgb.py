# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:20:40 2018

@author: Yanc_wang
"""


import sys
import h5py
import numpy as np
import xgboost as xgb
import scipy.io as scio
###----- data pre-processing -----###
data = scio.loadmat('D:\\py_ex\\xgbooost\\f.mat')

#data2= h5py.File('test.mat')
samTrain = data["samTrain"]
samTest = data["samTest"]
labTrain = data["lab_trn"]
labTest = data["lab_tst"]
dtrain = xgb.DMatrix(samTrain,labTrain-1)
dtest = xgb.DMatrix(samTest)
#print(data1.keys())
#train_x = data1['fv_trn'][:].transpose((1,0))
#train_y = data1['fv_tst'][:].transpose((1,0))
#test_x = data1['label_trn'][:].transpose((1,0))
#test_y = data1['label_tst'][:].transpose((1,0))

#train_x=np.vstack((pos_data_train,neg_data_train))
#train_y=np.hstack((np.ones(np.shape(pos_data_train)[0]),np.zeros(np.shape(neg_data_train)[0])))

#test_x=np.vstack((pos_data_test,neg_data_test))
#test_y=np.hstack((np.ones(np.shape(pos_data_test)[0]),np.zeros(np.shape(neg_data_test)[0])))


#dtest1=xgb.DMatrix(neg_data_test)


#----- Xgboost -----#
param = {
         'booster': 'gbtree',
         'gamma':0.4,
         'subsample':0.8,
         'colsample_bytree':0.8,
         'max_depth':12, 
         'lambda': 3,
         'eta':0.1, 
         'min_child_weight': 3,
         'seed':2000,
         'silent':1, 
         'objective':'multi:softmax',
         'num_class': 10,
         'nthread':4
         } 
#  params = {'max_depth':5,'eta':0.1,'objective':'binary:logistic','min_child_weight':1,
#  'gamma':0,'subsample':0.8,'colsample_bytree':0.9,'seed':0,'alpha':0.01,'lambda':0.1}
#  params['nthread'] = 4 
#  params['silent']=1
#evallist = [(dtest,'eval'), (dtrain,'train')]
num_round = 10
bst = xgb.train( param, dtrain, num_round)

ypred = bst.predict(dtest)
# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(labTest)):
    if ypred[i]+1 == labTest[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))


#model = xgb.XGBClassifier(
#        learning_rate=0.1,
#         min_child_weight=1,
#         gamma = 0,
#         subsample = 0.8,
#         colsample_bytree = 0.8,
#         max_depth = 5, 
#         eta = 1, 
#         scale_pos_weight = 1,
#         seed = 0,
#         silent = 1, 
##         objective='binary:logistic',
#         nthread = 4)
#model.fit(samTrain, labTrain)
#ypred1 = model.predict(dtest)           # ypred --输出概率


# =============================================================================

