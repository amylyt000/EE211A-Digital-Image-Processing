import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from matplotlib.colors import LinearSegmentedColormap
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
###This is the code we use 10-fold crossvalidation to test our model
def get_data_per_pixel(data):
    height, width = data.shape
    new_data = np.zeros([height, 62+60])
    for i in np.arange(height):
        for j in np.arange(0, 62):
            new_data[i,j] = data[i, 9*j+4]
        for j in np.arange(0, 60):
            new_data[i,j+62] = data[i, 558+j]
    return new_data

cdict1 = {
    'red':   ((0.0, 0.0, 0.0),
              (0.35, 0.0, 0.0),
              (0.66, 1.0, 1.0),
              (0.89, 1.0, 1.0),
              (1.0, 0.5, 0.5)),

    'green': ((0.0, 0.0, 0.0),
              (0.125, 0, 0),
              (0.375, 1, 1),
              (0.64, 1, 1),
              (0.91, 0, 0),
              (1, 0, 0)
             ),

    'blue':   ((0.0, 0.0, 0.0),
              (0.11, 1.0, 1.0),
              (0.34, 1, 1),
              (0.65, 0.5, 0.5),
              (1.0, 0.0, 0.0)),
         }

train_path = "./data/train_data_final_50k.csv"
df_train = pd.read_csv(train_path, header = None, index_col=False)
df_train = np.array(df_train)

X = np.asarray(df_train[:, 4:-1], dtype = np.float32)
y = np.asarray(df_train[:, -1], dtype = np.int32) #512^2

train_data = np.asarray(get_data_per_pixel(X), dtype=np.float32)
train_labels = np.asarray(y, dtype=np.int32)
print(train_data.shape)
scoring = ['average_precision','roc_auc','f1']#three score will be calculate during the process
#classifier:logisticregression(C=0.01),sgdclassifier(loss="log")
classifier = SVC(probability=True)
scores = cross_validate(classifier, train_data, train_labels, scoring=scoring,cv=10, return_train_score=False)
print("fit_time",np.mean(scores['fit_time']))
print("average precision",np.mean(scores['test_average_precision']))
print("average precision std",np.std(scores['test_average_precision']))
print("roc_auc",np.mean(scores['test_roc_auc']))
print("roc_auc std",np.std(scores['test_roc_auc']))
print("f1",np.mean(scores['test_f1']))
print("f1 std",np.std(scores['test_f1']))
