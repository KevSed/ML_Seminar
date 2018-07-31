import os
from tqdm import tqdm
from skimage.transform import resize
import imageio
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
import multiprocessing
from keras.utils import np_utils


def get_data(Folder_name, outfile):
    X = []
    Y = []
    for foldername in os.listdir(Folder_name):
        if not foldername.startswith('.'):
            if foldername in ['NORMAL']:
                label = 0
            if foldername in ['CNV']:
                label = 1
            if foldername in ['DME']:
                label = 2
            if foldername in ['DRUSEN']:
                label = 3
            counter = 0
            for image_filename in tqdm(os.listdir(Folder_name + foldername)):
                if(image_filename != '.DS_Store'):
                    img_file = imageio.imread(Folder_name+foldername+'/'+image_filename)
                    if img_file is not None:
                        img_file = resize(img_file,(50,100))
                        Y.append(label)
                        counter +=1
                        X.append(np.asarray(img_file))
    data = {}
    data['X_Train'] = X
    data['Y_Train'] = Y
    outputfile = h5py.File(outfile, mode='w')
    outputfile.create_dataset("X_Train", data=[data['X_Train'][i] for i in range(len(X))])
    outputfile.create_dataset("Y_Train", data=[data['Y_Train'][i] for i in range(len(Y))])
    outputfile.close()

    return X,Y



def input_for_training(file, xname, yname):
    inputfile = h5py.File(file, mode='r')
    X = inputfile[xname]
    Y = inputfile[yname]
    return X,Y

def weights(X, Y):
    labels_dist = np.bincount(Y)
    w = np.ones(len(Y), dtype=float)
    for s in range(len(Y)):
        w[s]*=1./labels_dist[Y[s]]
    return w



def scanning(X):
    means = []

    for sy in range(0,len(X),2):
        for sx in range(0,len(X[0]),4):
            sum = 0
            for dsy in range(2):
                sum+=np.sum(X[(sy+dsy)][sx:sx+4])
            means.append(sum/8)
    means = means/np.amax(means)
    return means

def plot_image(X, label, suffix):
    plt.figure()
    plt.imshow(X)
    plt.title(label)
    plt.savefig('/home/bjoern/Studium/ML/ML_Seminar/Alternative/TestOutput/Images/'+suffix+'.pdf')
    plt.close()

def plot_dist(X, label, suffix):
    plt.figure()
    plt.plot(X,  label='Means of (2,4) window')
    plt.title(label)
    plt.legend(loc='best')
    plt.savefig('/home/bjoern/Studium/ML/ML_Seminar/Alternative/TestOutput/Distribution/'+suffix+'.pdf')
    plt.close()



def example_plots():
    X,Y = get_data('/home/bjoern/Studium/ML/OCT2017/val/', 'val.hdf5')

    name = ['NORMAL', 'CNV', 'DME', 'DRUSEN']

    for i in range(len(X)):
        plot_image(X[i], name[Y[i]], name[Y[i]]+'/image_'+str(i))


    pool = multiprocessing.Pool()

    X = np.asarray(pool.map(scanning, tqdm(X)))

    for i in range(len(X)):
        plot_dist(X[i], name[Y[i]], name[Y[i]]+'/image_'+str(i))




def prep_eval(file, outfile):

    hdf5_file = h5py.File(file, mode='r')

    X_pre = hdf5_file['X_Train']

    Y_train = hdf5_file['Y_Train']

    pool = multiprocessing.Pool()

    X_train = np.asarray(pool.map(scanning, tqdm(X_pre)))

    Y_train = np.asarray(Y_train)

    X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_train, Y_train, test_size=0.25, random_state=43, stratify=Y_train, shuffle=True)

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Train, Y_Train, test_size=0.10, random_state=43, stratify=Y_Train, shuffle=True)

    weights_Train = weights(X_Train, Y_Train)
    weights_Val = weights(X_Val, Y_Val)*0.25/(0.75*0.9)
    weights_Test = weights(X_Test, Y_Test)
    Y_Train = np_utils.to_categorical(Y_Train,4)
    Y_Val = np_utils.to_categorical(Y_Val,4)
    Y_Test = np_utils.to_categorical(Y_Test, 4)


    data = {}
    data['X_Train'] = X_Train
    data['Y_Train'] = Y_Train
    data['weights_Train'] = weights_Train
    data['X_Val'] = X_Val
    data['Y_Val'] = Y_Val
    data['weights_Val'] = weights_Val
    data['X_Test'] = X_Test
    data['Y_Test'] = Y_Test
    data['weights_Test'] = weights_Test

    outputfile = h5py.File(outfile, mode='w')
    outputfile.create_dataset("X_Train", data=[data['X_Train'][i] for i in range(len(X_Train))])
    outputfile.create_dataset("Y_Train", data=[data['Y_Train'][i] for i in range(len(Y_Train))])
    outputfile.create_dataset("weights_Train", data=[data['weights_Train'][i] for i in range(len(weights_Train))])
    outputfile.create_dataset("X_Val", data=[data['X_Val'][i] for i in range(len(X_Val))])
    outputfile.create_dataset("Y_Val", data=[data['Y_Val'][i] for i in range(len(Y_Val))])
    outputfile.create_dataset("weights_Val", data=[data['weights_Val'][i] for i in range(len(weights_Val))])
    outputfile.create_dataset("Y_Test", data=[data['Y_Test'][i] for i in range(len(Y_Test))])
    outputfile.create_dataset("X_Test", data=[data['X_Test'][i] for i in range(len(X_Test))])
    outputfile.create_dataset("weights_Test", data=[data['weights_Test'][i] for i in range(len(weights_Test))])
    outputfile.close()
