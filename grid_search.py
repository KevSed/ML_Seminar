from sklearn.model_selection import GridSearchCV
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier


def make_model(dense_layer_sizes, filters,
               kernel_size, pool_size, stride_size, dropout_rate):
    '''Creates model comprised of 2 convolutional layers followed by dense layers

    dense_layer_sizes: List of layer sizes. This list has one number for each layer
    dense_activation: activation funciton in dense layer
    filters: Number of convolutional filters in each convolutional layer
    kernel_size: Convolutional kernel size
    pool_size: Size of pooling area for max pooling
    padding_type: type of padding: same or valid
    stride_size: symmetric stride size
    dropout_rate: dropout rate
    optimizer: optimizer used for mimizing
    '''

    model = Sequential()

    model.add(Conv2D(filters[0], kernel_size=(kernel_size[0], kernel_size[0]), padding='valid', activation='elu', strides=(stride_size[0], stride_size[0]), input_shape=shape_ord, kernel_initializer='VarianceScaling'))
    model.add(MaxPooling2D(pool_size=(pool_size[0], pool_size[0]), padding='valid'))
    model.add(Conv2D(filters[1], kernel_size=(kernel_size[1], kernel_size[1]), padding='valid', activation='elu', strides=(stride_size[1], stride_size[1])))
    model.add(MaxPooling2D(pool_size=(pool_size[1], pool_size[1]), padding='valid'))
    # model.add(Flatten())
    model.add(Dropout(dropout_rate[0]))
    model.add(Dense(dense_layer_sizes[0], activation='elu'))
    model.add(Dense(dense_layer_sizes[1], activation='elu'))
    model.add(Flatten())
    model.add(Dense(dense_layer_sizes[2], activation='elu'))
    model.add(Dropout(dropout_rate[1]))
    model.add(Dense(dense_layer_sizes[3], activation='elu'))
   # model.add(Dense(16, activation='relu'))
   # model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def main():

    image_files = h5py.File('whole_image_set.hdf5')
    data = image_files['train_img']
    labels = image_files['train_label']

    X_train, X_test, Y_train, Y_test = train_test_split(data[:], labels[:], test_size=0., shuffle=True, stratify=labels)
    X_train = X_train.reshape(len(X_train), img_rows**2)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)

    if K.image_data_format() == 'channels_first':
        shape_ord = (1, img_rows, img_cols)
    else:  # channel_last
        shape_ord = (img_rows, img_cols, 1)

    X_train = X_train.reshape((X_train.shape[0],) + shape_ord)
    Y_train = np_utils.to_categorical(Y_train, 4)

    my_cnn = KerasClassifier(make_model)

    dense_size_candidates = [[i for i in range(500, 1500, 100)], [i for i in range(125, 375, 25)], [i for i in range(50, 150, 10)], [i for i in range(16, 48, 2)], [4]]
    filters = [[i for i in range(32, 128, 8)], [i for i in range(8, 64, 8)]]
    kernel_size = [[2, 4], [2, 4]]
    pool_size = [[2, 3, 4, 5], [2, 3, 4, 5]]
    stride_size = [[1, 2, 3, 4], [1, 2, 3, 4]]
    dropout_rate = [[0.25, 0.5, 0.75], [0.25, 0.5, 0.75]]
    batch_size = [k for k in range(100, 500, 50)]
    epochs = [e for e in range(10, 50, 5)]

    param_grid={'dense_layer_sizes': dense_size_candidates,
                 'filters': filters,
                 'kernel_size': kernel_size,
                 'pool_size': pool_size,
                 'stride_size'  : stride_size,
                 'dropout_rate' : dropout_rate,
                 'epochs': epochs,
                 'batch_size': batch_size
                }


    filepath = "best_cnn.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

    grid = GridSearchCV(my_cnn, param_grid, cv=2, scoring='average_precision', n_jobs=48)
    grid_result = grid.fit(X_train, Y_train, callbacks=[checkpoint])

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == '__main__':
    main()
