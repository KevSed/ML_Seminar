import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
import matplotlib.pyplot as plt
import itertools
nichTrue = False
img_rows, img_cols = 400, 400

def plot_confusion_matrix(cm, classes, what,
                          normalize=nichTrue,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    matplotlib.rcParams.update({'font.size': 18})
    plt.title(title + ' ' + what)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, ['NORMAL', 'CNV', 'DME', 'DRUSEN'], rotation=45)
    plt.yticks(tick_marks, ['NORMAL', 'CNV', 'DME', 'DRUSEN'])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('pdf/confusion_matrix_{}.pdf'.format(what))
    plt.close()


def evaluate(X_val, Y_val, model, weights, what):

    klassen = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
    ##Evaluate loss and metrics
    print('Evaluating model...')
    loss, accuracy = model.evaluate(X_val, Y_val, sample_weight=weights, verbose= 1)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)
    print('Predicting on validation data...')
    # Predict the values from the test dataset
    Y_pred = model.predict(X_val, verbose=1)
    # Convert predictions classes to one hot vectors
    Y_cls = np.argmax(Y_pred, axis = 1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_val, axis = 1)
    print('Classification Report:\n', classification_report(Y_true,Y_cls))

    ## Plot 0 probability
    for label in range(4):
        Y_pred_prob = Y_pred[:,label]

        weights_t = np.ones(len(Y_pred_prob[Y_true == label])) / len(Y_pred_prob[Y_true == label])
        weights_f = np.ones(len(Y_pred_prob[Y_true != label])) / len(Y_pred_prob[Y_true != label])

        plt.figure()
        plt.hist(Y_pred_prob[Y_true == label], weights=weights_t, alpha=0.5, color='red', bins=20, range=(0,1), log = True, label=klassen[label])
        plt.hist(Y_pred_prob[Y_true != label], weights=weights_f, alpha=0.5, color='blue', bins=20, range=(0,1), log = True, label='!={}'.format(klassen[label]))
        plt.legend(loc='best')
        plt.xlabel('Probability of being {}'.format(klassen[label]))
        plt.ylabel('Number of entries')
        plt.savefig('pdf/{}_or_not_log_{}.pdf'.format(klassen[label], what))
        plt.close()
        plt.hist(Y_pred_prob[Y_true == label], weights=weights_t, alpha=0.5, color='red', bins=20, range=(0,1), label=klassen[label])
        plt.hist(Y_pred_prob[Y_true != label], weights=weights_f , alpha=0.5, color='blue', bins=20, range=(0,1), label='!={}'.format(klassen[label]))
        plt.legend(loc='best')
        plt.xlabel('Probability of being {}'.format(klassen[label]))
        plt.ylabel('Number of entries')
        plt.savefig('pdf/{}_or_not_{}.pdf'.format(klassen[label], what))
        plt.close()
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_cls, labels=range(4))
    # plot the confusion matrix
    plt.figure(figsize=(8,8))
    plot_confusion_matrix(confusion_mtx, range(4), what, normalize=True)
