import numpy as np
import tensorflow as tf
import pandas as pd
import os
import glob
import re
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Permute, Reshape, Dropout, BatchNormalization
from keras.layers import Conv1D, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


# def get_unique_data(data):
#     height, width = data.shape
#     new_data = np.zeros([height, 62+60])
#     for i in np.arange(height):
#         for j in np.arange(0, 62):
#             new_data[i,j] = data[i, 9*j+4]
#         for j in np.arange(0, 60):
#             new_data[i,j+62] = data[i, 558+j]
#     return new_data



def reduce_data_dim(data):
    # 512^2 * 618 to 256^2 *618
    # reduce_data = np.zeros([256**2, 618])
    trans_data = np.reshape(data.transpose((1, 0)), (618, 512, 512))
    # print(trans_data.shape)
    crop_data = trans_data[:, 128:384, 128:384]
    # print(crop_data.shape)
    reduce_data = np.transpose(crop_data.reshape((618, 256**2)), (1,0))

    return reduce_data

def get_stack_data(data):
    # 256^2 * 618 to 256^2 * 3* (62*3) * 1
    height = data.shape[0]
    stack_data = np.zeros([height, 3, 62*3])
    for i in range(height):
        # crop_this_row = data[i, 0:(62*9)]
        for j in range(62):
            get_nine_pixel = data[i, 9*j:9*(j+1)]
            three_by_three = np.reshape(get_nine_pixel, (3,3))
            # print(three_by_three.shape)
            stack_data[i, :, 3*j:3*(j+1)] = three_by_three
    return stack_data



def get_train_data():
    data_folder = r"./data/test_data/test_data"
    label_folder = r"./data/labels"
    img_folder = r"./data/imgs"
    data_files = glob.glob(data_folder + "/*.csv")
    num_images = 10
    # train_x = np.zeros([NUM_DATA_FILE * TRAIN_SIZE, 22, 1000,1])
    train_data = np.zeros([num_images*65536, 3, 3*62, 1])
    train_labels = np.zeros([65536*num_images]) # row vector
    i = 0
    flag = True
    for filename in data_files:
        # print(filename)
        if flag == True:
            (filepath,tempfilename) = os.path.split(filename)
            (shortname,extension) = os.path.splitext(tempfilename)
            patient_id = shortname.split('-')[0]
            slide = shortname.split('-')[-1]
            match = re.match(r"([a-z]+)([0-9]+)", slide, re.I)
            slide_id = match.group(2)
            
        
            if os.path.isfile(label_folder + "/" + shortname + '-label.csv') == True:
                if os.path.isfile(img_folder + "/" + patient_id + "_original_slice_" + slide_id + '.jpg') == False:
                    if i<num_images:
                        # print(i)
                        tr_data  = pd.read_csv(filename, header = None, index_col=False)
                        tr_data = np.asarray(tr_data, dtype = np.float32)
                    
                        if tr_data.shape[0] == 512**2:
                            # print(i)
                            # print(tr_data.shape)
                            tr_data = reduce_data_dim(tr_data) # 512^2 * 618 to 256^2 * 618
                            stack_data = get_stack_data(tr_data) # 256^2 * 618 to 256^2 * 3 *(62*3)
                            stack_data = np.reshape(stack_data, (256**2, 3, 186, 1))
                            # tr_data = tr_data.reshape((65536,1,618,1))
                            # print(tr_data.shape)
                            train_data[i*65536:(i+1)*65536, :, :, :] = stack_data
                            tr_labels = pd.read_csv(label_folder + "/" + shortname + '-label.csv', header = None, index_col=False)
                            tr_labels = np.asarray(tr_labels, dtype = np.int32)
                           
                            tr_labels = tr_labels.reshape((512, 512))
                            # print(tr_labels.shape)
                            tr_labels = tr_labels[128:384, 128:384]
                            # print(tr_labels.shape)
                            tr_labels = tr_labels.reshape((1, 65536))
                            train_labels[65536*i:65536*(i+1)] = tr_labels
                            i += 1
                        elif tr_data.shape[0] == 256**2:
                            # print(i)
                            stack_data = get_stack_data(tr_data)
                            stack_data = np.reshape(stack_data, (256**2, 3, 186, 1))
                            train_data[i*65536:(i+1)*65536, :, :, :] = stack_data
                            tr_labels = pd.read_csv(label_folder + "/" + shortname + '-label.csv', header = None, index_col=False)
                            tr_labels = np.asarray(tr_labels, dtype = np.int32)
                            train_labels[65536*i:65536*(i+1)] = tr_labels.reshape((1, -1))
                            i += 1
                    else:
                        flag = False

    return train_data, train_labels


def cnn_model():
    print('==================================================================================================')
    model = Sequential()
    # 1st conv layer
    
    model.add(Conv2D(20, (1, 5), padding='valid', input_shape = (3, 186, 1)))

    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))


    model.add(Conv2D(20, (3, 1), padding = 'valid'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Permute((3,2,1)))
    model.add(MaxPooling2D(pool_size = (1,2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(50, (20,10), padding = 'valid'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Permute((3,2,1)))
    model.add(MaxPooling2D(pool_size = (1,2)))
  

    model.add(Conv2D(100, (50,2), padding = 'valid'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Permute((3,2,1)))
    # model.add(MaxPooling2D(pool_size = (1,2)))

    model.add(Reshape((100*40,)))

    model.add(Dense(2))

    print(model.layers[-1].output.shape)

    model.add(Activation('softmax'))


    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def train_cnn(model, train_x,train_y,validation_split  = 0.2):
    callbacks_list = [ModelCheckpoint('model.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='auto', save_weights_only='True')]
    model.fit(train_x, train_y, batch_size=100, epochs=200, verbose=2, callbacks=callbacks_list,validation_split = validation_split)
    
    return model

def test(model):
    test_data_folder = r"./data/test_data/test_data"
    test_label_folder = r"./data/labels"
    test_img_folder = r"./data/imgs"
    test_data_files = glob.glob(test_data_folder + "/*.csv")

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


    for filename in test_data_files:
        (filepath,tempfilename) = os.path.split(filename)
        (shortname,extension) = os.path.splitext(tempfilename)
        patient_id = shortname.split('-')[0]
        slide = shortname.split('-')[-1]
        match = re.match(r"([a-z]+)([0-9]+)", slide, re.I)
        slide_id = match.group(2)

        if os.path.exists(test_img_folder + "/" + patient_id + "_original_slice_" + slide_id + '.jpg') == True:
            test_data = pd.read_csv(filename, header = None, index_col=False)
            test_data = np.asarray(test_data, dtype = np.float32)
            if test_data.shape[0]  == 256**2:
                test_data = np.reshape(get_stack_data(test_data), (256**2, 3, 186, 1))
            elif test_data.shape[0] == 512**2:
                # test_data = reduce_data_dim(test_data) 
                test_data = get_stack_data(test_data) 
                test_data = np.reshape(test_data, (512**2, 3, 186, 1))

            test_img_path = test_img_folder + "/" + patient_id + "_original_slice_" + slide_id + '.jpg'
            print(test_img_path)
            print(test_data.shape)
            # probability of being predicted as 1
            predictions = model.predict(test_data)
            print(predictions.shape)
            # y_prob = [p["probabilities"][1] for p in predictions]

            plt.figure(figsize=(15,15))
            test_pred = predictions[:, 1]
            print(test_pred.shape)
            test_pred_reshape = np.reshape(test_pred, (int(np.sqrt(len(test_pred))), -1))

            test_img = plt.imread(test_img_path)

            revise_jet = LinearSegmentedColormap('revise_jet', cdict1)
            revise_img = np.rot90(test_pred_reshape)*255
            revise_img = np.array(revise_img)
            canvas = np.where(revise_img<20)
            revise_img[canvas] = 0
            plt.imshow(revise_img, cmap=revise_jet)
            plt.savefig('./predict_cnn/' + patient_id + '_original_slice_' + slide_id + '.png')
                # plt.show()

    print("Done!")


if __name__ == "__main__":
    print("step1")
    train_data, train_labels = get_train_data()
    print(train_data.shape)
    print(train_labels.shape)

    print("step2")
    cnn = cnn_model()

    print("step3")
    print(train_data.shape)
    # cnn.fit(cnn, train_data, train_labels )
    model = train_cnn(cnn, train_data, train_labels)

    print("step4")
    test(model)


