import numpy as np
from numpy import *
import pandas as pd
from numpy.fft import *
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def process(data):
    test = data
    indexs = test[:, 0]
    bubbles = []
    indexs = indexs.tolist()
    num = 13

    X = []
    Y = []
    Z = []
    for j in range(num):
        tmp_x = []
        tmp_y = []
        tmp_z = []
        for i in range(len(test)) :
            if j+1 == indexs[i] :
                tmp_x.append(test[i,1])
                tmp_y.append(test[i,2])
                tmp_z.append(test[i,3])
        X.append(tmp_x)
        Y.append(tmp_y)
        Z.append(tmp_z)

    k_means = KMeans(n_clusters=num)
    k_means.fit(test[:,1:3])

    print(adjusted_rand_score(indexs, k_means.predict(test[:,1:3])))

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(num):
        ax.scatter(X[i],Y[i],Z[i], marker='o', label = i)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

if __name__ == '__main__':
    process()

