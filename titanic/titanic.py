#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import os

# from tabulate import tabulate

from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn import preprocessing, neighbors, svm

from sklearn.externals import joblib
import time

# import random
#
# random.seed(30)

# In[2]:


# plt.style.use('ggplot')

# In[3]:

titanic_base_dir_path = 'D:/Py/My_DS/titanic'
titanic_data_dir_path = os.path.join(titanic_base_dir_path, 'titanic')
titanic_file = 'gender_submission.csv'
titanic_train_f_name = 'train.csv'
titanic_test_f_name = 'test.csv'

df_y = pd.read_csv(os.path.join(titanic_data_dir_path, titanic_test_f_name))

import pickle

# clf = svm.SVC(kernel='linear', cache_size=7000, probability=False)
#
# for i, batch_name in enumerate([file_name for file_name in os.listdir(titanic_data_dir_path) if
#                                 file_name[0:len(
#                                     'data_batch_')] == 'data_batch_' and file_name != 'data_batch_21_size_32.pickle']):
#     print(batch_name)
#     data = pickle.load(
#         open(os.path.join(titanic_data_dir_path, batch_name), 'rb'))
#     start_time = round(time.time(), 0)
#     print('start {}th run, on {}.'.format(i + 1, start_time))
#     # print(data[0], data[1])
#     clf.fit(data[0], data[1])
#     joblib.dump(clf, os.path.join(titanic_base_dir_path, 'models',
#                                   'svm_linear_titanic{}_on_{}_with_batch_{}.pkl'.format(i + 1, time.time(),
#                                                                                         32)))
#     end_time = round(time.time(), 0)
#     print('end {}th run in {}.'.format(i + 1, end_time - start_time))
#     print(''.join(['@'] * 20))
clf = joblib.load(
    os.path.join(titanic_base_dir_path, 'models', 'svm_linear_titanic27_on_1588436398.2479577_with_batch_32.pkl'))

x_test = pickle.load(open(os.path.join(titanic_data_dir_path, 'data_test.pickle'), 'rb'))
x_test = pd.DataFrame(x_test)
# print(x_test.info())
ids = x_test['PassengerId']
x_test.drop(['PassengerId'], 1, inplace=True)
predictions = pd.DataFrame(clf.predict(x_test), columns=['Survived'])
# print(ids)
# print(predictions)
# ids['Survived'] = predictions
predictions = predictions.set_index([pd.Series(ids)])
print(predictions)
predictions.to_csv(os.path.join(titanic_base_dir_path, 'prediction_PL.csv'))

# data_batch_21_size_32.pickle
