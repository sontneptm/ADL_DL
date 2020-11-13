import tensorflow as tf
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, GRU, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

acc_data = np.loadtxt('6axis_aae_tf2.csv', delimiter=',') 

x_data = acc_data[:, 1:]
y_data = acc_data[:, :1]

enc = OneHotEncoder()
enc.fit(y_data)
yt_onehot = enc.transform(y_data).toarray()
y_data = yt_onehot

x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))
xt, xv, yt, yv,  = train_test_split(x_data, y_data, test_size=0.2, shuffle= True)

model = Sequential()
model.add(LSTM(128, activation=tf.nn.swish, input_shape=(xt.shape[1], 1)))
#model.add(Dense(1024, activation=tf.nn.swish, input_shape=[xt.shape[1]]))
model.add(Dense(1024, activation=tf.nn.swish))
model.add(Dense(1024, activation=tf.nn.swish))
model.add(Dense(1024, activation=tf.nn.swish))
model.add(Dense(13, activation='softmax'))
model.summary()

model.compile(optimizer=Adam(lr=1.46e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(xt, yt, validation_split=0.2, epochs= 3)

test_loss, test_acc = model.evaluate(xv, yv, verbose=2)
print(test_acc)
pd = model.predict(xv)
yv = yv.argmax(axis=1)
pd = pd.argmax(axis=1)

print(confusion_matrix(yv, pd))
print(classification_report(yv, pd, digits=4))