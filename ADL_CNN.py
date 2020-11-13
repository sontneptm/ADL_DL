import os
import glob
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import random
from PIL import Image
from pyarrow import csv

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

acc_data = np.loadtxt('papam2_aae_basic.csv', delimiter=',') 

x_data = acc_data[:, 1:]
y_data = acc_data[:, :1]

DATA_LEN = len(x_data[0])
EPOCH = 1000
SPLIT_RATE = 0.2

enc = OneHotEncoder()
enc.fit(y_data)
yt_onehot = enc.transform(y_data).toarray()
y_data = yt_onehot

x_data = x_data.reshape(-1,DATA_LEN,1)
xt, xv, yt, yv,  = train_test_split(x_data, y_data, test_size=0.2, shuffle= True)

print("done")

with tf.device('GPU:0'):
    model = models.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=2, activation=tf.nn.swish, input_shape=(DATA_LEN, 1))) 
    model.add(MaxPooling1D(pool_size=2))
    #model.add(layers.Conv1D(filters=32, kernel_size=2, activation=tf.nn.swish)) 
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(layers.Flatten())
    model.add(Dense(500, activation=tf.nn.swish))
    model.add(Dense(500, activation=tf.nn.swish))
    model.add(layers.Dense(12, activation='softmax'))

    model.summary()
    model.compile(optimizer=Adam(lr=1.46e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(xt, yt, epochs=100)

    test_loss, test_acc = model.evaluate(xv, yv, verbose=2)
    print(test_acc)