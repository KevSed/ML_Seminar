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
      pixels within a defined window. By default this window has the dimension (2,4).
      Furthermore, it performs train_test and train_val splits and saves all
      datasets to outfile. To take the different class sizes into account weights
      are calculated and saved in outfile as well.
* grid.py: Contains the class grid. A class member has to be initialized with a
  dataset obtained from prep_eval containing training and validation datasets and
  their corresponding weightsself. By default it contains a string in Path_base containing the
  path to the folder where model structures to test are saved and a string in Out_file
  containing the path to the folder where fitted model structures, histories and
  trained weights are saved and have to be set correctly by the user before calling a method
  of this class.
    - make_model(self, dense_layer, activation, dropouts, out_activation, number)
      receives a network structure to be tested and saves the structure in a json
      file. The user has to assign a number to this model which is saved in the
      suffix of this file
    - fit_model(self, number, outnumber, outfile): Fits model corresponding to number and
      saves model and history of the training in Out_file as .json file and trained weights as .hdf5 file.
      The user has to assign a outnumber to this run. Loss and accuravy histories are saved in outfile.
* performance.py: Contains methods to evaluate the performance of models and
  to select models from grid search.
    - model_evaluator(mod, infiles, outfiles, lab, unblind=False): receives number corresponding to
      fitted model and the folder containing the corresponding json file for the model structure
      and hdf5 files for trained weights in infiles.
      Confusion matrix is plotted, classification_report is printed and the
      NN output for a given label correspondig to a analyzed class.
      This label is defined by lab. Outputs are saved to outfiles.
      If unblind is set to True the Performance will be evluated on the test dataset,
      otherwise on the validation dataset
    - model_selector(infiles, batch_size, tested_models, acc_thr,  loss_thr, outfile)
      selects models tested in grid search passing the two step selection.
      A model has to pass a minimum validation accuracy threshold, defined by the train accuracy - validation accuracy,
      and a loss threshold defined by the loss function after all epochs being smaller than loss_thr*loss function
      after 1 epoch. It returns an array with all numbers corresponding to models
      passing the selection. tested_models correspond to the tested number of
      layer structures (This is very hard coded, sorry for that)
    - model_plotter(infiles,outfiles, batch_size, tested_models) plots for all
      tested layer structures the validation accuracy and loss after all epochs
      as function of the tested batch size given in batch_size and tested
      acitvation functions for the hidden layers and output layer.
      Outputs are saved to out files.
* A complete example of the analysis is given below. Note that it is necessary to download the Dataset first:
  https://www.kaggle.com/paultimothymooney/kermany2018. Example_plots shows images after resizing and input distributions for the DNN
  The hdf5 file will be 3.3GB large, if you use the train dataset in the OCT2017 folder


"""

Path = '/home/bjoern/Studium/ML'

example_plots()

get_data(Path + '/OCT2017/test/', 'test.hdf5')




prep_eval('test.hdf5', 'evaluate_test.hdf5')

dataset = h5py.File('evaluate_test.hdf5', mode='r')


X = grid(dataset)

X.Path_Base = Path + '/ML_Seminar/Alternative/TestOutput/ModelBase/'
X.Out_file = Path + '/ML_Seminar/Alternative/TestOutput/Files/'
# Tested models

dense_layer =[[1024,512,128,64,32], [1024, 512,256,128,64,32,16], [512,256,128,64,32,16], [1024, 256, 64, 16], [512, 128, 32]]
activation = ['relu', 'elu']
out_activation = ['softmax', 'sigmoid']
dropouts = [[0.5, 0.4,0.4, 0.3, 0.2], [0.5, 0.4, 0.4, 0.4, 0.2, 0.2, 0.1], [0.4, 0.4, 0.3, 0.3, 0.2, 0.1], [0.6, 0.4, 0.2, 0.1], [0.5, 0.3, 0.1]]


tested_models = []
number = 0
for i in range(len(dense_layer)):
    for acti in activation:
        for out_acti in out_activation:
            X.make_model(dense_layer[i], acti, dropouts[i], out_acti, number)
            tested_models.append(number)
            number += 1
batch_size = [50, 64,100, 128, 256, 512]

for i in tested_models:
    for b in range(len(batch_size)):
        X.batch_size = batch_size[b]
        X.fit_model(i, i*len(batch_size)+b, Path + '/ML_Seminar/Alternative/TestOutput/Output/')




models = model_selector(Path + '/ML_Seminar/Alternative/TestOutput/Files/', [50,64,100,128,256,512], 5, 0.20, 1, Path + '/ML_Seminar/Alternative/TestOutput/AfterSel/')

model_plotter(Path + '/ML_Seminar/Alternative/TestOutput/Files/', Path + '/ML_Seminar/Alternative/TestOutput/ModelEval/',[50,64,100,128,256,512],5)
print(models)
accuracy = []
for i in models:
    accuracy.append(model_evaluator(i, 'evaluate_test.hdf5',Path + '/ML_Seminar/Alternative/TestOutput/Files/',Path + '/ML_Seminar/Alternative/TestOutput/Performance/',0, unblind=False ))

# Spoiler ALERT: This is the model which wins the test !!!
model_evaluator(6,'evaluate_test.hdf5', Path + '/ML_Seminar/Alternative/TestOutput/Files/',Path + '/ML_Seminar/Alternative/TestOutput/PerformanceTest/',0, unblind=True )

print(accuracy)

# Select the model with the highest accuracy
