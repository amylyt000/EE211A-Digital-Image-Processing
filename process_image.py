import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from PIL import Image

orig_folder = r'./prediction/original'
cnn_folder = r'./prediction/cnn'
logis_folder = r'./prediction/logis'
sgd_svm_folder = r'./prediction/sgdmodifiedhuber'
svm_folder = r'./prediction/svm'
sgd_logis_folder = r'./prediction/predict_sgd_log'

orig_images = glob.glob(orig_folder + '/*.jpg')
cnn_images = glob.glob(cnn_folder + '/*.png')
logis_images = glob.glob(logis_folder + '/*.png')
sgd_svm_images = glob.glob(sgd_svm_folder + '/*.png')
svm_images = glob.glob(svm_folder + '/*.png')
sgd_logis_images = glob.glob(sgd_logis_folder + '/*.png')
# num_images_per_model = 7
# num_models = 4


orig_resize_folder = r'./prediction/orig_resize'
orig_resize_images = glob.glob(orig_resize_folder + '/*.png')

i = 0
# concat_img_orig = np.ones([1200, 7980])

for each_path in orig_resize_images:
    img = plt.imread(each_path)
    print(i)

    if i == 0:
      # concat_img_orig = img
        concat_img_orig = np.concatenate((img, np.ones([1500, 156])), axis = 1)
      # print(concat_img.shape)
    elif i == 6:
        concat_img_orig = np.concatenate((concat_img_orig, img), axis = 1)
    else:
        concat_img_orig = np.concatenate((concat_img_orig, img), axis = 1)
        concat_img_orig = np.concatenate((concat_img_orig, np.ones([1500, 156])), axis = 1)

    i+=1

concat_grey = plt.imshow(concat_img_orig, cmap = plt.get_cmap('gray'))
plt.axis('off')
plt.savefig('./concat_grey.png')

i = 0
for each_path in cnn_images:
    img = plt.imread(each_path)
    
    img = img[150:1350, 150:1350]
    # print(img.shape)
    if i == 0:
        concat_img_cnn = img
        # print(concat_img.shape)
    else:
        concat_img_cnn = np.concatenate((concat_img_cnn, img), axis = 1)
    i += 1
print(concat_img_cnn.shape)

i = 0
for each_path in logis_images:
    img = plt.imread(each_path)
    img = img[150:1350, 150:1350]
    if i == 0:
        concat_img_logis = img
        # print(concat_img.shape)
    else:
        concat_img_logis = np.concatenate((concat_img_logis, img), axis = 1)
    i += 1

i = 0
for each_path in sgd_svm_images:
    img = plt.imread(each_path)
    img = img[150:1350, 150:1350]
    if i == 0:
        concat_img_sgd = img
        # print(concat_img.shape)
    else:
        concat_img_sgd = np.concatenate((concat_img_sgd, img), axis = 1)
    i += 1

i = 0
for each_path in svm_images:
    img = plt.imread(each_path)
    img = img[150:1350, 150:1350]
    if i == 0:
        concat_img_svm = img
        # print(concat_img.shape)
    else:
        concat_img_svm = np.concatenate((concat_img_svm, img), axis = 1)
    i += 1

i = 0
for each_path in sgd_logis_images:
    img = plt.imread(each_path)
    img = img[150:1350, 150:1350]
    if i == 0:
        concat_img_sgd_logis = img
        # print(concat_img.shape)
    else:
        concat_img_sgd_logis = np.concatenate((concat_img_sgd_logis, img), axis = 1)
    i += 1    

concat_image = np.concatenate((concat_img_cnn, concat_img_logis, concat_img_sgd, concat_img_svm, concat_img_sgd_logis))
plt.figure()
plt.axis('off')
plt.imshow(concat_image, interpolation = 'nearest')
plt.savefig('./concat_images_withou.png')
# plt.show()

