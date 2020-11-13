import os
import glob
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dropout
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random
from PIL import Image
from pyarrow import csv

print("loading images... ", end='')
image_list = glob.glob('images/*.png')

labels=[]
images=[]
split_rate = 0.2

for i in image_list:
    label = int(i[i.index('_')+1:i.index('.')])-1
    """
    if label >= 3 and label <= 10 :
        continue 
    if label == 11:
        label =3
    if label == 12:
        label =4
    """ # for using only basis activities
    labels.append(label)
    im = Image.open(i)
    images.append(np.array(im))

train_img, test_img, train_label, test_label = train_test_split(
    images, labels, test_size = split_rate, random_state = 123)

train_img = np.array(train_img)
test_img = np.array(test_img)
train_label = np.array(train_label)
test_label = np.array(test_label)

train_img = train_img.reshape((-1,128,3,1))
test_img = test_img.reshape((-1,128,3,1))

train_img, test_img = train_img / 255.0, test_img / 255.0

print("done")

with tf.device('GPU:0'):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding ='same', activation='relu', input_shape=(128, 3, 1))) 
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(layers.Conv2D(128, (5, 3), padding ='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(layers.Conv2D(128, (7, 3), padding ='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(13, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_img, train_label, epochs=100)

    test_loss, test_acc = model.evaluate(test_img, test_label, verbose=2)
    print(test_acc)