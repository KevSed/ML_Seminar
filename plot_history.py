import matplotlib.pyplot as plt
import numpy as np


def plot_history(loss, val_loss, acc, val_acc, name):
    """
    Plots the accuracy and loss history for the Training of a model.

    Inputs:
    -----------------------------------------------------------------
    loss:       loss per epoch of training
    val_loss:   loss per epoch of validation
    acc:        accuracy per epoch of training
    val_acc:    accuracy per epoch of validation
    name:       string to determine model

    Returns:
    -----------------------------------------------------------------
    None. Saves histories to files
    """

    # Weights for 70/30 splitting
    a = 0.7
    b = (1 - a)
    a *=  10**4
    b *=  10**4
    if name in 'equal':
        a = 1
        b = 1
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss*(np.ones(len(loss))*a))
    plt.plot(val_loss*(np.ones(len(loss))*b))
    plt.title('Loss history')
    plt.legend(['Training', 'Validation'])
    plt.savefig('pdf/loss_history_{}.pdf'.format(name))
    plt.close()

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Accuracy history')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig('pdf/accuracy_history_{}.pdf'.format(name))
    plt.close()
