import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
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
        plt.hist(Y_pred_prob[Y_true == label], alpha=0.5, color='red', bins=20, range=(0,1), log = True, density = True)
        plt.hist(Y_pred_prob[Y_true != label], alpha=0.5, color='blue', bins=20, range=(0,1), log = True, density = True)
        plt.legend(['NORMAL', klassen[label]], loc='upper right')
        plt.xlabel('Probability of being {}'.format(klassen[label]))
        plt.ylabel('Number of entries')
        plt.savefig('pdf/{}_or_not_log_{}.pdf'.format(klassen[label], what))
        plt.close()
        plt.hist(Y_pred_prob[Y_true == label], alpha=0.5, color='red',  bins=20, range=(0,1), log = nichTrue, density = True)
        plt.hist(Y_pred_prob[Y_true != label], alpha=0.5, color='blue', bins=20, range=(0,1), log = nichTrue, density = True)
        plt.legend(['NORMAL', klassen[label]], loc='upper right')
        plt.xlabel('Probability of being {}'.format(klassen[label]))
        plt.ylabel('Number of entries')
        plt.savefig('pdf/{}_or_not_{}.pdf'.format(klassen[label], what))
        plt.close()
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_cls, labels=range(4))
    # plot the confusion matrix
    plt.figure(figsize=(8,8))
    plot_confusion_matrix(confusion_mtx, range(4), what, normalize=True)

    # #Plot largest errors
    # errors = (Y_cls - Y_true != 0)
    # Y_cls_errors = Y_cls[errors]
    # Y_pred_errors = Y_pred[errors]
    # Y_true_errors = Y_true[errors]
    # X_val_errors = X_val[errors]
    # # Probabilities of the wrong predicted numbers
    # Y_pred_errors_prob = np.max(Y_pred_errors, axis = 1)
    # # Predicted probabilities of the true values in the error set
    # true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
    # # Difference between the probability of the predicted label and the true label
    # delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
    # # Sorted list of the delta prob errors
    # sorted_dela_errors = np.argsort(delta_pred_true_errors)
    # # Top 6 errors
    # most_important_errors = sorted_dela_errors[-6:]
    # # Show the top 6 errors
    # # display_errors(most_important_errors, X_val_errors, Y_cls_errors, Y_true_errors)
    #
    # ##Plot predictions
    # slice = 10
    # predicted = model.predict(X_val[:slice]).argmax(-1)
    # plt.figure(figsize=(16,8))
    # for i in range(slice):
    #     plt.subplot(1, slice, i+1)
    #     plt.imshow(X_val[i].reshape(img_rows, img_cols), interpolation='nearest')
    #     plt.title('True: {}'.format(klassen[Y_val[i]]))
    #     plt.text(0, 0, predicted[i], color='black',
    #              bbox=dict(facecolor='white', alpha=1))
    #     plt.axis('off')
    # plt.savefig('pdf/predictions_{}.pdf'.format(what))
