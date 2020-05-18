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

df_test = pd.read_csv(os.path.join(titanic_data_dir_path, titanic_test_f_name))
df_train = pd.read_csv(os.path.join(titanic_data_dir_path, titanic_train_f_name))
df_y = pd.read_csv(os.path.join(titanic_data_dir_path, titanic_file))

# formatting matter, joining test and train
df = pd.concat([df_train, df_test])

# print('-' * 100)
df.info()
# preprocessing
original_df = pd.DataFrame.copy(df)
# filling missing data
df.drop(['Name'], 1, inplace=True)
df.fillna(9999, inplace=True)
# changeing values at float
df = df.apply(
    lambda col: pd.factorize(col)[0].astype(np.float64)
    if col.dtype not in ['int64', 'float64'] else col)

# print('-'*100)
# df_y.info()
# print('-'*100)
# df.info()


# In[4]:


# fare and tickets are connected so this is not true fare
# df.sort_values('Ticket').head(10)


# In[5]:


df_tmp = df.groupby('Ticket').agg(['count'])
df_tmp = df_tmp.iloc[:, 0:1]
df_tmp.columns = ['count']
df = df.merge(df_tmp, on='Ticket', how='left')
# df.head()


# In[6]:


# calculating true fare and droping columns
df['true_fare'] = round(df['Fare'] / df['count'], 2)
df.drop(['count', 'Ticket', 'Fare', 'Cabin'], 1, inplace=True)

# In[7]:


# making dummies
df = pd.get_dummies(df, columns=['Embarked', 'Sex', 'Pclass'], dtype=float)
# df.head()


# In[36]:


# last preprocessing
x_train = df[df['Survived'] < 2]
x_test = df[df['Survived'] == 9999.0]
y_train = x_train['Survived']
x_train.drop(['PassengerId', 'Survived'], 1, inplace=True)
x_test.drop(['Survived'], 1, inplace=True)

import pickle

# In[ ]:

batch_size = 32


# print(len(x_train))

#
def get_chunk(x_data, y_data, batch_size):
    for i in range(0, len(x_data), batch_size):
        yield [x_data[i:i + batch_size], y_data[i:i + batch_size]]


# clf = svm.SVC(kernel='linear', cache_size=7000, probability=False)

# for i, data in enumerate(get_chunk(x_train, y_train, batch_size)):
#     start_time = round(time.time(), 0)
#     print('start {}th run, on {}.'.format(i + 1, start_time))
#     # print(data[0], data[1])
#     # clf.fit(data[0], data[1])
#     pickle.dump(data,
#                 open(os.path.join(titanic_data_dir_path, 'data_batch_{}_size_{}.pickle'.format(i + 1, batch_size)),
#                      'wb'))
#     # joblib.dump(clf, os.path.join(titanic_base_dir_path,'models',
#     #                               'svm_linear_titanic{}_on_{}_with_batch_{}.pkl'.format(i + 1, time.time(), batch_size)))
#     end_time = round(time.time(), 0)
#     print('end {}th run in {}.'.format(i + 1, end_time - start_time))
#     print(''.join(['@'] * 20))

pickle.dump(x_test, open(os.path.join(titanic_data_dir_path, 'data_test.pickle'), 'wb'))
# accurycy = clf.score(x_test, df_y['Survived'].to_numpy())
# print(accurycy)
