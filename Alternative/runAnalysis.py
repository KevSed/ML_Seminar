from dataprep import *
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
import h5py
from keras import optimizers
from sklearn.metrics import confusion_matrix,classification_report
import itertools
from performance import *
from grid import *


"""
These are important things to read!!!

This is the file where you should put everything what you want to run.
* dataprep.py: Contains methods for data preparation:
    - get_data(Folder, File) reads all images from Folder and saves it to File.
      This File should be a hdf5 file, which contains the pixel arrays of each
      image in File['X_Train'] and labels in File['Y_Train']. By default images
      are resized to (50,100) shape.
    - prep_eval(file, outfile) reads in file obtained by get_data(Folder,File)
      and performs the image scan. The image scan calculates the means of all
      the pixels within a defined window. By default this is set to (2,4) and
      are taken as features of scanned imagesself.
      Furthermore, it performs train_test and train_val splits and saves all
      sets to outfile. To take the different class sizes into account weights
      are calculated and saved in outfile as well.
* grid.py: Contains the class grid. A class member has to be initialized with a
  dataset obtained from prep_eval. By default it contains a string in Path_base containing the
  path to the folder where model structures to test are saved and a string in Out_file
  containing the path to the folder where fitted model structures, histories and
  trained weights are saved and have to be set correctly by the user before calling a method
  of this class.
    - make_model(self, dense_layer, activation, dropouts, out_activation, number)
      receives a network structure to be tested and saves the structure in a json
      file. The user has to assign a number to this model which is saved in the
      suffix of this file
    - fit_model(self, number, outnumber, outfile): Fits model corresponding to number and
      saves model, trained weights and history of the training. The user has to
      assign a outnumber to this run. Loss and accuravy histories are saved in outfile.
* performance.py: Contains methods to evaluate the performance of models and
  to select models from grid search.
    - model_evaluator(mod, infiles, outfiles, lab): receives number corresponding to
      fitted model and the folder containing the corresponding json file in infiles.
      Confusion matrix is plotted, classification_report is printed and the
      NN output for a given label. This label is defined by lab. Outputs are saved to
      outfiles.
    - model_selector(infiles, batch_size, tested_models, acc_thr, ovt_thr, loss_thr)
      selects models tested in grid search passing the three step selection.
      A model has to pass a minimum validation accuracy threshold, an overtraining threshold
      defined by the train accuracy - validation accuracy and a loss threshold defined
      by the loss function after all epochs being smaller than loss_thr*loss function
      after 1 epoch. It returns an array with all numbers corresponding to model
      passing the selection. tested_models correspond to the tested number of
      layer structures
    - model_plotter(infiles,outfiles, batch_size, tested_models) plots for all
      tested layer structures the validation accuracy and loss after all epochs
      as function of the tested batch sizes given in batch_size and tested
      acitvation functions for the hidden layers and output layer.
      Outputs are saved to out files.

"""

#get_data('OCT2017/train/', 'train.hdf5')

#prep_eval('train.hdf5', 'evaluate.hdf5')

#dataset = h5py.File('evaluate.hdf5', mode='r')

"""
X = grid(dataset)
X.batch_size = 50


dense_layer =[[1024,512,128,64,32]]
activation = ['relu']
out_activation = ['sigmoid']
dropouts = [[0.5, 0.4,0.4, 0.3, 0.2]]



number = 0
for i in range(len(dense_layer)):
    for acti in activation:
        for out_acti in out_activation:
            X.make_model(dense_layer[i], acti, dropouts[i], out_acti, number)
            number += 1

X.fit_model(0, 1)
"""

print(model_selector('/home/bjoern/Studium/ML/GridSearch/Files/', [50,64,100,128,256,512], 5, 0.73, 0.05, 0.95))

model_plotter('/home/bjoern/Studium/ML/GridSearch/Files/', 'GridSearch/ModelEval/',[50,64,100,128,256,512],5 )
model_evaluator(6, '/home/bjoern/Studium/ML/GridSearch/Files/','GridSearch/Performance/',0 )



"""
dense_layer =[[1024,512,128,64,32], [1024, 512,256,128,64,32,16], [512,256,128,64,32,16], [1024, 256, 64, 16], [512, 128, 32]]
activation = ['relu', 'elu']
out_activation = ['softmax', 'sigmoid']
dropouts = [[0.5, 0.4,0.4, 0.3, 0.2], [0.5, 0.4, 0.4, 0.4, 0.2, 0.2, 0.1], [0.4, 0.4, 0.3, 0.3, 0.2, 0.1], [0.6, 0.4, 0.2, 0.1], [0.5, 0.3, 0.1]]
input_dim = 625
"""
