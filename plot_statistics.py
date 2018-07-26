import h5py
from keras.models import load_model
from plot_history import plot_history
from evaluate import evaluate
import click


@click.command()
@click.argument('name')
def main(name):
    print('Plotting statistics for Architecture:', name)
    print('Loading history...')
    h = h5py.File('history_{}.h5'.format(name))
    print('Loading weights and validation data...')
    v = h5py.File('val_weights_{}.h5'.format(name))
    print('Loading model...')
    m = load_model('model_{}.hdf5'.format(name))

    print('plot loss and accuracy history')
    plot_history(h['loss'], h['val_loss'], h['accuracy'], h['val_accuracy'], name)

    print('plot confusion matrix, ill or not plot')
    evaluate(v['X_val'], v['Y_val'], m, v['w_val'], name)

if __name__ == '__main__':
    main()
