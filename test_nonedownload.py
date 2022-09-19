import time
from multiprocessing.pool import ThreadPool
import requests
import os
import PIL.Image as Image
from io import BytesIO
import cv2
import numpy as np
from download import *
import pandas as pd



def loadData(fname):
    data = []
    f = open(fname, 'r+', encoding='utf-8')
    count = 0
    while count<200000:
        line = f.readline()
        line = line.rstrip('\n')
        data.append(line)
        count += 1
        if not line:
            break
    f.close()

    data_clean1 = []
    for i in data:
        if len(i) >230:
            data_clean1.append(i)

    data_clean = []
    for i in data_clean1:
        if i[0] != '!':
            data_clean.append(i)
    return data_clean


###########################   main   ###################
def download_image(url):
    url = loadData(fname='./ImageLinksFinal.txt')
    image = download_image_thread(url, our_dir='./image', num_processes=16, remove_bad=False, Async=True)
    # image = []
    # for i in image1:
    #     image.append(i.get())
    np.save("image.npy", image)
    # image = np.load("./image.npy")

# preprocess, delete the none row
url = loadData(fname='./ImageLinksFinal.txt')
image = download_image_thread(url[0:20], our_dir='./image', num_processes=32, remove_bad=False, Async=True)
ind = []
for i in range(len(image)):
    if image[i].get() is None:
        ind.append(i)
print(ind)
indd = pd.DataFrame(data=ind)
indd.to_csv("./index_none.csv")



