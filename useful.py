# coding: utf-8
import matplotlib.pyplot as plt
import itertools
import matplotlib.cm as cm
from keras import backend as K
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas
import h5py
import imageio
import os
from skimage.transform import resize
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import click
from multiprocessing import Pool
from time import sleep


img_rows, img_cols = 500, 500
np.random.seed(1338)  # for reproducibilty


def get_data(folderName, folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    if not folderName.startswith('.'):
        if folderName in ['NORMAL']:
            label = 0
        elif folderName in ['CNV']:
            label = 1
        elif folderName in ['DME']:
            label = 2
        elif folderName in ['DRUSEN']:
            label = 3
        print("Loading files from directory ", folderName)
        for image_filename in os.listdir(folder + folderName):
            if image_filename.endswith('.jpeg'):
                img_file = imageio.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = resize(img_file, (img_rows, img_cols))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

def as_completed(futures):
    futures = list(futures)
    while futures:
        for f in futures.copy():
            if f.ready():
                futures.remove(f)
                yield f.get()
        sleep(0.1)

FOLDER = "../OCT2017/train/"

data = []
labels = []
gesamt = 0

