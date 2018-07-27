import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
import h5py
from keras import optimizers
from sklearn.metrics import confusion_matrix,classification_report
import itertools


def plot_confusion_matrix(cm, d,outfile, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    sum = 0
    for i in range(len(cm)):
        sum += cm[i][i]
    print('Corrected accuracy: '+str(0.25*sum)+'!!!!!')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(outfile+'confusion_matrix_'+str(d)+'.pdf')
    plt.close()
    return 0.25*sum

def evaluate(X_test, Y_test, weights_Test,model,d,lab, outfile):
    name = ['NORMAL', 'CNV','DME', 'DRUSEN']
    not_name = ['KRANK', 'NOT CNV', 'NOT DME', 'NOT DRUSEN']
    ##Evaluate loss and metrics
    loss, accuracy = model.evaluate(X_test, Y_test, sample_weight=weights_Test, verbose=0)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)
    # Predict the values from the test dataset
    Y_pred = model.predict(X_test)
    # Convert predictions classes to one hot vectors
    Y_cls = np.argmax(Y_pred, axis = 1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_test, axis = 1)
    print('Classification Report:\n', classification_report(Y_true,Y_cls))

    ## Plot 0 probability
    label=lab
    Y_pred_prob = Y_pred[:,label]
    norm_fact = np.ones(len(Y_pred_prob[Y_true == label]), dtype=float)*1./len(Y_pred_prob[Y_true == label])
    norm_fact2 = np.ones(len(Y_pred_prob[Y_true != label]), dtype=float)*1./len(Y_pred_prob[Y_true != label])
    plt.hist(Y_pred_prob[Y_true == label], alpha=0.5,weights=norm_fact, color='red', bins=20, range=(0., 1.), log = True)
    plt.hist(Y_pred_prob[Y_true != label], alpha=0.5,weights=norm_fact2, color='blue', bins=20, range=(0., 1.), log = True)
    plt.legend([name[lab], not_name[lab]], loc='upper right')
    plt.xlabel('Probability of being '+name[lab])
    plt.ylabel('Number of entries')
    plt.savefig(outfile+name[lab]+'_or_not_log_'+str(d)+'.pdf')
    plt.close()

    plt.hist(Y_pred_prob[Y_true == label], alpha=0.5,weights=norm_fact, color='red',  bins=20,range=(0., 1.), log = False)
    plt.hist(Y_pred_prob[Y_true != label], alpha=0.5,weights=norm_fact2, color='blue', bins=20,range=(0., 1.), log = False)
    plt.legend([name[lab], not_name[lab]], loc='upper right')
    plt.xlabel('Probability of being '+name[lab])
    plt.ylabel('Number of entries')
    plt.savefig(outfile+name[lab]+'_or_not_'+str(d)+'.pdf')
    plt.close()

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_cls)

    # plot the confusion matrix
    plt.figure(figsize=(8,8))
    corr_acc = plot_confusion_matrix(confusion_mtx, d,outfile, classes = range(4))
    return corr_acc

def model_selector(infiles, batch_size, tested_models, acc_thr, ovt_thr, loss_thr):
    #Path = '/home/bjoern/Studium/ML/GridSearch/Files/'
    Path = infiles
    surviving = []
    for i in range(tested_models*4):
            for k in range(len(batch_size)):
                s = i*len(batch_size)+k
                history = h5py.File(Path+'hist_'+str(s)+'.hdf5', mode='r')
                val_acc = history['val_acc'][149]
                acc = history['acc'][149]
                loss_start = history['val_loss'][0]
                loss_end = history['val_loss'][149]
                if(val_acc < acc_thr):
                    print('Model '+str(int(s/6))+' fails accury selection for batch size ' + str(batch_size[k]))
                    continue
                if(val_acc < acc-ovt_thr):
                    print('Model '+str(int(s/6))+' fails overtraining selection for batch size ' + str(batch_size[k]))
                    continue
                if(loss_end > loss_thr*loss_start):
                    print('Model '+str(int(s/6))+' fails loss selection for batch size ' + str(batch_size[k]))
                    continue
                surviving.append(s)
    return surviving


def model_evaluator(mod, infiles, outfiles, lab):

    Path = infiles
    dataset = h5py.File('evaluate.hdf5', mode='r')
    X_Val = dataset['X_Val']
    Y_Val = dataset['Y_Val']
    weights_Val = dataset['weights_Val']
    print('Processing model '+str(mod))
    json_file = open(Path+'model_'+str(mod)+'.json', 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(Path+'model_'+str(mod)+'.hdf5')
    model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])
    loss, accuracy = model.evaluate(X_Val, Y_Val, sample_weight=weights_Val, verbose=0)
    print('Val accuracy:', accuracy)
    #for lab in [0,1,2,3]:
    #    print('Corrected accuracy: '+str(evaluate(X_Val, Y_Val, weights_Val, model, mod,lab)))
    return evaluate(X_Val, Y_Val, weights_Val, model, mod,lab, outfiles)

def model_plotter(infiles,outfiles, batch_size, tested_models):

    Path = infiles
    for i in range(tested_models):
        softmax_relu = []
        sigmoid_relu = []
        softmax_elu = []
        sigmoid_elu = []
        softmax_relu_val_loss =[]
        sigmoid_relu_val_loss = []
        softmax_elu_val_loss =[]
        sigmoid_elu_val_loss = []
        for s in range(4):
            for k in range(len(batch_size)):
                mod = (4*i)*len(batch_size)+s*len(batch_size)+k
                history = h5py.File(Path+'hist_'+str(mod)+'.hdf5', mode='r')
                val_acc = history['val_acc'][-1]
                acc = history['acc'][-1]
                loss_start = history['val_loss'][0]
                loss_end = history['val_loss'][-1]
                if(s==0):
                    softmax_relu.append(val_acc)
                    softmax_relu_val_loss.append(loss_end*10**4)
                elif(s==1):
                    sigmoid_relu.append(val_acc)
                    sigmoid_relu_val_loss.append(loss_end*10**4)
                elif(s==2):
                    softmax_elu.append(val_acc)
                    softmax_elu_val_loss.append(loss_end*10**4)
                elif(s==3):
                    sigmoid_elu.append(val_acc)
                    sigmoid_elu_val_loss.append(loss_end*10**4)
        plt.figure()
        plt.plot(batch_size, softmax_relu_val_loss, 'rx', label='Softmax relu')
        plt.plot(batch_size, sigmoid_relu_val_loss, 'bx', label='Sigmoid relu')
        plt.plot(batch_size, softmax_elu_val_loss,'kx', label='Softmax elu')
        plt.plot(batch_size, sigmoid_elu_val_loss,'gx', label='Sigmoid elu')
        plt.xlabel('Batch size')
        plt.ylabel(r'Validation loss / $10^{4}$')
        plt.title('Model '+str(i))
        plt.legend(loc='best')

        plt.savefig(outfiles+'Model_val_loss'+str(i)+'.pdf')
        plt.close()

        plt.figure()
        plt.plot(batch_size, softmax_relu, 'rx', label='Softmax relu')
        plt.plot(batch_size, sigmoid_relu, 'bx', label='Sigmoid relu')
        plt.plot(batch_size, softmax_elu,'kx', label='Softmax elu')
        plt.plot(batch_size, sigmoid_elu,'gx', label='Sigmoid elu')
        plt.xlabel('Batch size')
        plt.ylabel('Validation accuracy')
        plt.title('Model '+str(i))
        plt.legend(loc='best')

        plt.savefig(outfiles+'Model_val_acc'+str(i)+'.pdf')
        plt.close()
