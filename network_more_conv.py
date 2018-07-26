# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import h5py
import imageio
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from evaluate import evaluate
from plot_history import plot_history
from save_all import save_all


img_rows, img_cols = 400, 400
np.random.seed(1338) # for reproducibilty


def weights(Y):
    labels_dist = np.bincount(Y)
    w = np.ones(len(Y), dtype=float)
    for s in range(len(Y)):
        w[s]*=1./labels_dist[Y[s]]
    return w


def main():

    image_files = h5py.File('whole_image_set.hdf5')
    data = image_files['train_img']
    labels = image_files['train_label']


    X_train, X_val, Y_train, Y_val = train_test_split(data[:], labels[:], test_size=0.3, stratify=labels, shuffle=True)


    print('''
    Datensätze:
    ------------------------------------
    \t \t train \t val \t shape
    Data:\t {} \t {} \t {}
    Labels: \t {} \t {} \t {}
    '''.format(len(X_train), len(X_val), X_train.shape, len(Y_train), len(Y_val), Y_train.shape))

    X_train = X_train.reshape(len(X_train), img_rows**2)
    X_val = X_val.reshape(len(X_val), img_rows**2)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    cp = sns.countplot(Y_train)
    fig = cp.get_figure()
    fig.savefig('pdf/countplot_new.pdf')
    fig.clf()

    if K.image_data_format() == 'channels_first':
        shape_ord = (1, img_rows, img_cols)
    else:  # channel_last
        shape_ord = (img_rows, img_cols, 1)

    X_train = X_train.reshape((X_train.shape[0],) + shape_ord)
    X_val = X_val.reshape((X_val.shape[0],) + shape_ord)

    labels_dist = np.bincount(Y_train)

    w_train = weights(Y_train)
    w_val = weights(Y_val)

    Y_train = np_utils.to_categorical(Y_train, 4)
    Y_val = np_utils.to_categorical(Y_val, 4)


    print('''
    \t\t Y \t\t X
    ---------------------------------------------------------------
    train|\t {} \t {}
    test |\t {} \t {}
    '''.format(Y_train.shape, X_train.shape, Y_val.shape, X_val.shape))

    # -- Initializing the values for the convolution neural network

    nb_epoch = 40
    batch_size = 100
    model = Sequential()
    model.add(Conv2D(512, kernel_size=(4, 4), padding='valid', activation='elu', strides=(2, 2), input_shape=shape_ord, kernel_initializer='VarianceScaling'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=(4, 4), padding='valid', activation='elu', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(3, 3), padding='valid'))
    model.add(Conv2D(32, kernel_size=(2, 2), padding='valid', activation='elu', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dense(256, activation='elu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(4, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("\n\t Fitting model \n")
    hist = model.fit(X_train, Y_train, batch_size=batch_size,
                 epochs=nb_epoch, verbose=1, sample_weight=w_train,
                 validation_data=(X_val, Y_val, w_val))

    model.save('./model_new.hdf5')
    save_all(hist, X_val, Y_val, w_val, 'new')


if __name__ == '__main__':
    main()
