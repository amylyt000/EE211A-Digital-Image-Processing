import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.metrics
from sklearn.cluster import KNN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("final_d??.csv")
X = df[[]]
y = df[[]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # train:test = 5:1



