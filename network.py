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
from .evaluate import evaluate
from .plot_history import plot_history


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.pdf')
    plt.close()


def evaluate(X_val, Y_val, model):

    ##Evaluate loss and metrics
    loss, accuracy = model.evaluate(X_val, Y_val, verbose=0)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)
    # Predict the values from the test dataset
    Y_pred = model.predict(X_val)
    # Convert predictions classes to one hot vectors
    Y_cls = np.argmax(Y_pred, axis = 1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_val, axis = 1)
    print('Classification Report:\n', classification_report(Y_true,Y_cls))

    ## Plot 0 probability
    label=0
    Y_pred_prob = Y_pred[:,label]
    plt.hist(Y_pred_prob[Y_true == label], alpha=0.5, color='red', bins=10, log = True)
    plt.hist(Y_pred_prob[Y_true != label], alpha=0.5, color='blue', bins=10, log = True)
    plt.legend(['NORMAL', 'KRANK'], loc='upper right')
    plt.xlabel('Probability of being 0')
    plt.ylabel('Number of entries')
    plt.savefig('ill_or_not.pdf')
    plt.close()

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_cls)
    # plot the confusion matrix
    plt.figure(figsize=(8,8))
    plot_confusion_matrix(confusion_mtx, classes = range(4))

    #Plot largest errors
    errors = (Y_cls - Y_true != 0)
    Y_cls_errors = Y_cls[errors]
    Y_pred_errors = Y_pred[errors]
    Y_true_errors = Y_true[errors]
    X_val_errors = X_val[errors]
    # Probabilities of the wrong predicted numbers
    Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
    # Predicted probabilities of the true values in the error set
    true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
    # Difference between the probability of the predicted label and the true label
    delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
    # Sorted list of the delta prob errors
    sorted_dela_errors = np.argsort(delta_pred_true_errors)
    # Top 6 errors
    most_important_errors = sorted_dela_errors[-6:]
    # Show the top 6 errors
    # display_errors(most_important_errors, X_val_errors, Y_cls_errors, Y_true_errors)

    ##Plot predictions
    slice = 10
    predicted = model.predict(X_val[:slice]).argmax(-1)
    plt.figure(figsize=(16,8))
    for i in range(slice):
        plt.subplot(1, slice, i+1)
        plt.imshow(X_val[i].reshape(img_rows, img_cols), interpolation='nearest')
        plt.text(0, 0, predicted[i], color='black',
                 bbox=dict(facecolor='white', alpha=1))
        plt.axis('off')
    plt.savefig('predictions.pdf')


img_rows, img_cols = 400, 400
np.random.seed(1338)  # for reproducibilty


def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.savefig('loss_history.pdf')
    plt.close()

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history.history['acc'])
    plt.plot(network_history.history['val_acc'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig('accuracy_history.pdf')
    plt.close()




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
    fig.savefig('countplot.pdf')
    fig.clf()

    if K.image_data_format() == 'channels_first':
        shape_ord = (1, img_rows, img_cols)
    else:  # channel_last
        shape_ord = (img_rows, img_cols, 1)

    X_train = X_train.reshape((X_train.shape[0],) + shape_ord)
    X_val = X_val.reshape((X_val.shape[0],) + shape_ord)

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
    # kept very low!
    batch_size = 100

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(4, 4), padding='valid', activation='elu', strides=(2, 2), input_shape=shape_ord, kernel_initializer='VarianceScaling'))
    model.add(MaxPooling2D(pool_size=(3, 3), padding='valid'))
    model.add(Conv2D(32, kernel_size=(4, 4), padding='valid', activation='elu', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(3, 3), padding='valid'))
    # model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1000, activation='elu'))
    model.add(Dense(250, activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='elu'))
   # model.add(Dense(16, activation='relu'))
   # model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    print("\n\t Fitting model \n")
    hist = model.fit(X_train, Y_train, batch_size=batch_size,
                 epochs=nb_epoch, verbose=1,
                 validation_data=(X_val, Y_val))

    plot_history(hist)
    evaluate(X_val, Y_val, model)



if __name__ == '__main__':
    main()
