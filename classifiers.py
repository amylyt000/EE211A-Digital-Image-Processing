import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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

train_path = "data/train_data_final_50k.csv"
df_train = pd.read_csv(train_path, header = None, index_col=False)
df_train = np.array(df_train)

X = np.asarray(df_train[:, 4:-1], dtype = np.float32)
y = np.asarray(df_train[:, -1], dtype = np.int32) #512^2

train_data = np.asarray(get_data_per_pixel(X), dtype=np.float32)
train_labels = np.asarray(y, dtype=np.int32)
print(train_data.shape)


# find roc auc score and pr score
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size = 0.2, random_state = 0)
classifier = linear_model.LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_val)
#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
print('roc_auc_score:', roc_auc_score(y_val, predictions))
accuracy_score(y_true = y_val, y_pred = predictions)
print('average_precision_score:', average_precision_score(y_val, predictions))
#Compute the F1 score, also known as balanced F-score or F-measure
print('f1_score: ', f1_score(y_val, predictions, average=None))


clf = linear_model.LogisticRegression()
clf.fit(train_data, train_labels)

test_data_folder = r"./data/test_data/test_data"
test_label_folder = r"./data/labels"
test_img_folder = r"./data/imgs"
test_data_files = glob.glob(test_data_folder + "/*.csv")
# print(test_data_files)
for filename in test_data_files:

    (filepath,tempfilename) = os.path.split(filename)
    (shortname,extension) = os.path.splitext(tempfilename)
    patient_id = shortname.split('-')[0]
    slide = shortname.split('-')[-1]
    match = re.match(r"([a-z]+)([0-9]+)", slide, re.I)
    slide_id = match.group(2)
    
    if os.path.exists(test_img_folder + "/" + patient_id + "_original_slice_" + slide_id + '.jpg'):
        print(filename)
        test_data = pd.read_csv(filename, header = None, index_col=False)
        # test_img_path = test_img_folder + "/" + patient_id + "_original_slice_" + slide_id + '.jpg'
        test_data = np.asarray(test_data, dtype = np.float32)
        test_data = get_data_per_pixel(test_data)
        test_pred = clf.predict_proba(test_data)
        print(len(test_pred))
        test_pred_reshape = np.reshape(test_pred[:,1], (int(np.sqrt(len(test_pred))), -1))
        plt.figure(figsize=(15,15))
        revise_jet = LinearSegmentedColormap('revise_jet', cdict1)
        revise_img = np.rot90(test_pred_reshape)*255
        revise_img = np.array(revise_img)
        canvas = np.where(revise_img<70)
        revise_img[canvas] = 70
        plt.imshow(revise_img, cmap=revise_jet)
        plt.savefig('./predict_logis/'+patient_id + '_original_slice_' + slide_id + '.png')


