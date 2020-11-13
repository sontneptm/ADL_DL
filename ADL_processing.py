import numpy as np
from numpy import *
import pandas as pd
import time
import random
from PIL import Image

def six_axis_to_image():
    print("loading text...", end='')
    p = np.loadtxt('6axis_raw.csv', delimiter=',', dtype=np.float32)
    print("done")

    index = p[:,0:1]
    datas = p[:,1:]

    print("normalizing datas...", end='')

    for i in range(len(datas)):
        for j in range(len(datas[i])):
            if datas[i][j] >= 1.0:
                datas[i][j] = 1.0
            elif datas[i][j] <= -1.0:
                datas[i][j] = -1.0
   

    data_mean = datas.mean()
    tmp_data = np.absolute(datas)
    print(tmp_data.mean())
    data_max = datas.max()
    data_min = datas.min()
    
    normalized_data = np.zeros((len(datas), len(datas[0])))

    for i in range(len(datas)):
        for j in range(len(datas[i])):
            normalized_data[i][j] = (datas[i][j]-data_mean)/(data_max - data_min) + 0.465
            
    normalized_data *=255

    print(normalized_data.max())
    print(normalized_data.min())

    print("done")

    total_len = len(p)
    height = len(datas[0])//3
    width = 3

    images = np.zeros((total_len, height, width))

    print("converting to image...", end='')
    
    for i in range(total_len):
        for j in range(height):
            images[i][j][0] = (normalized_data[i][j]) # for X axis
            images[i][j][1] = (normalized_data[i][j+128]) # for X axis
            images[i][j][2] = (normalized_data[i][j+256]) # for X axis

        if i%100 == 0:
            print("image ", i, " changed")
   
    print("done")

    for i in range(total_len):
        im = Image.fromarray(images[i])
        im = im.convert("L")
        im_url = 'images/image'+str(i)+"_"+str(int(index[i].tolist()[0]))+".png"
        im.save(im_url)
        if i%100 == 0:
            print("image ", i, " saved")
         

    print("done")

if __name__ == '__main__':
    six_axis_to_image()