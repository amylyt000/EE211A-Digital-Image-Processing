
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import glob
import re
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.INFO)




def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    input_layer = tf.reshape(features["x"], [-1, 1, 618, 1])

    # Convolutional Layer #1

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[1, 6],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[1, 6],
          padding="same",
          activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 3], strides=3)

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(inputs=pool2,
                             filters=64,
                             kernel_size= [1,4],
                             padding = "same",
                             activation=tf.nn.relu)
    # Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size = [1, 4], strides=3)

    # Flatten tensor into a batch of vectors
    pool3_flat = tf.reshape(pool3, [-1, 1 * 34 * 64])

    # Dense Layer
    dense = tf.layers.dense(inputs=pool3_flat, units=250, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    train_path = "./data/train_data_final_50k.csv"
    df_train = pd.read_csv(train_path, header = None, index_col=False)
    df = np.array(df_train)
    X = np.asarray(df[:, 4:-1], dtype = np.float32)
    y = np.asarray(df[:, -1], dtype = np.int32)

    train_data = np.asarray(X, dtype=np.float32)
    train_labels = np.asarray(y, dtype=np.int32)

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./convnet_model")
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=200,
        hooks=[logging_hook])


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
# ================================================================================================
    test_data_folder = r"./data/test_data/test_data"
    test_label_folder = r"./data/labels"
    test_img_folder = r"./data/imgs"
    test_data_files = glob.glob(test_data_folder + "/*.csv")

    test_predictions = {}

    for filename in test_data_files:
        (filepath,tempfilename) = os.path.split(filename)
        (shortname,extension) = os.path.splitext(tempfilename)
        patient_id = shortname.split('-')[0]
        slide = shortname.split('-')[-1]
        match = re.match(r"([a-z]+)([0-9]+)", slide, re.I)
        slide_id = match.group(2)


        test_data = pd.read_csv(filename, header = None, index_col=False)
        test_data = np.asarray(test_data, dtype = np.float32)

        test_img_path = test_img_folder + "/" + patient_id + "_original_slice_" + slide_id + '.jpg'
        print(test_img_path)


    # Evaluate the model and print results
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_data},
            y=None,
            num_epochs=1,
            shuffle=False)
    

        # # predicted class
        # predictions = classifier.predict(input_fn=eval_input_fn)
        # y_pred = [p["class_ids"][0] for p in predictions]

        # probability of being predicted as 1
        predictions = classifier.predict(input_fn=test_input_fn)
        y_prob = [p["probabilities"][1] for p in predictions]

        plt.figure(figsize=(15,15))
        test_pred = y_prob

        test_pred_reshape = np.reshape(test_pred, (int(np.sqrt(len(test_pred))), -1))


        test_img = plt.imread(test_img_path)
        revise_jet = LinearSegmentedColormap('revise_jet', cdict1)
        revise_img = np.rot90(test_pred_reshape)*255
        revise_img = np.array(revise_img)
        # print(revise_img.min())
        canvas = np.where(revise_img<20)
        revise_img[canvas] = 0
        plt.imshow(revise_img, cmap=revise_jet)
        plt.savefig('./predict_cnn_1D/'+patient_id + '_original_slice_' + slide_id + '.png')
        # plt.show()
    
    
        # test_predictions = {}
        test_predictions[test_img_path] = test_pred_reshape

    print("Done!")

if __name__ == "__main__":
    tf.app.run()
    




