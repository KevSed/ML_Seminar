import h5py
from keras.models import load_model
from plot_history import plot_history
from evaluate import evaluate
from sklearn.preprocessing import MinMaxScaler
import click
import numpy as np
import seaborn as sns
from keras import backend as K
from keras.utils import np_utils


def main():

    img_rows, img_cols = 400, 400
    image_files = h5py.File('upload_image_set.hdf5')
    X_val = np.asarray(image_files['train_img'])
    Y_val = np.asarray(image_files['train_label'])

    print('''
    Datensätze:
    ------------------------------------
    \t \t val \t shape
    Data:\t {} \t {}
    Labels:\t {} \t {}
    '''.format(len(X_val), X_val.shape, len(Y_val), Y_val.shape))

    X_val = X_val.reshape(len(X_val), img_rows**2)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_val = scaler.fit_transform(X_val)
    X_val = scaler.transform(X_val)

    cp = sns.countplot(Y_val)
    fig = cp.get_figure()
    fig.savefig('countplot.pdf')
    fig.clf()

    if K.image_data_format() == 'channels_first':
        shape_ord = (1, img_rows, img_cols)
    else:  # channel_last
        shape_ord = (img_rows, img_cols, 1)

    X_val = X_val.reshape((X_val.shape[0],) + shape_ord)

    Y_val = np_utils.to_categorical(Y_val, 4)

    print('Plotting statistics for Architecture:', 'CNN')
    print('Loading model...')
    m = load_model('model_smaller.hdf5')

    print('plot confusion matrix, ill or not plot')
    evaluate(X_val, Y_val, m, np.ones(len(Y_val)), 'CNN')

if __name__ == '__main__':
    main()
