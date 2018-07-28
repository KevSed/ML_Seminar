import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from keras import backend as K
from skimage.transform import resize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from array import array
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix,classification_report
from keras.regularizers import l2
from keras import optimizers
import h5py
import multiprocessing
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json


class grid:

    X_Val = np.asarray([])
    Y_Val = np.asarray([])
    weights_Val = np.asarray([])
    X_Train = np.asarray([])
    Y_Train = np.asarray([])
    weights_Train = np.asarray([])
    batch_size = 0
    Path_base = ''
    Out_file = ''

    def __init__(self, dictio):
        self.X_Val = np.asarray(dictio['X_Val'])
        self.Y_Val = np.asarray(dictio['Y_Val'])
        self.weights_Val = np.asarray(dictio['weights_Val'])
        self.X_Train = np.asarray(dictio['X_Train'])
        self.Y_Train = np.asarray(dictio['Y_Train'])
        self.weights_Train = np.asarray(dictio['weights_Train'])
        self.Path_base = '/home/bjoern/Studium/ML/GridSearch/ModelBase/'
        self.batch_size = 128
        print('Base models are saved to and read from /home/bjoern/Studium/ML/GridSearch/ModelBase/. Define a new directory via self.Path_Base if you want to save it elsewhere!')
        self.Out_file = '/home/bjoern/Studium/ML/GridSearch/Files/'
        print('Histories, trained weights and models are saved to and read from /home/bjoern/Studium/ML/GridSearch/Files/. Define a new directory via self.Out_Base if you want to save it elsewhere!')

    def make_model(self, dense_layer, activation, dropouts, out_activation, number):
        np.random.seed(1338)

        model = Sequential()
        for i in range(len(dense_layer)):
            if(i == 0):
                model.add(Dense(dense_layer[i],activation=activation, input_dim=len(self.X_Train)))
                model.add(Dropout(dropouts[i]))
                continue
            model.add(Dense(dense_layer[i],activation=activation))
            model.add(Dropout(dropouts[i]))
        model.add(Dense(4,activation=out_activation))
        #model.summary()
        model_json = model.to_json()
        with open(self.Path_base+'model_'+str(number)+'.json', "w") as json_file:
            json_file.write(model_json)


    def fit_model(self,number, outnumber, outfile):
        json_file = open(self.Path_base+'model_'+str(number)+'.json', 'r')
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.summary()
        model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])
        history = model.fit(self.X_Train, self.Y_Train, sample_weight=self.weights_Train, validation_data = (self.X_Val, self.Y_Val, self.weights_Val), epochs=150, batch_size=self.batch_size)
        model_json = model.to_json()
        with open(self.Out_file+'model_'+str(outnumber)+'.json', "w") as json_file:
            json_file.write(model_json)
        model.save_weights(self.Out_file+'model_'+str(outnumber)+'.hdf5')
        print("Saved model "+str(outnumber)+" to disk ")
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['Training', 'Validation'])
        plt.savefig(outfile+'loss_history_'+str(outnumber)+'.pdf')
        plt.close()

        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.savefig(outfile+'accuracy_history_'+str(outnumber)+'.pdf')
        plt.close()

        data = {}
        data['loss'] = history.history['loss']
        data['val_loss'] = history.history['val_loss']
        data['acc'] = history.history['acc']
        data['val_acc'] = history.history['val_acc']
        hdf5_file = h5py.File(self.Out_file+'hist_'+str(number)+'.hdf5', mode='w')
        hdf5_file.create_dataset("loss", data=[data['loss'][i] for i in range(len(data['loss']))])
        hdf5_file.create_dataset("val_loss", data=[data['val_loss'][i] for i in range(len(data['val_loss']))])
        hdf5_file.create_dataset("acc", data=[data['acc'][i] for i in range(len(data['acc']))])
        hdf5_file.create_dataset("val_acc", data=[data['val_acc'][i] for i in range(len(data['val_acc']))])
        hdf5_file.close()
