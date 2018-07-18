from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import plot_model
import matplotlib.image as mpimg
from keras import backend as K

img_rows, img_cols = 400, 400

if K.image_data_format() == 'channels_first':
    shape_ord = (1, img_rows, img_cols)
else:  # channel_last
    shape_ord = (img_rows, img_cols, 1)



model = Sequential()

model.add(Conv2D(64, kernel_size=(4, 4), padding='valid', activation='elu', strides=(2, 2), input_shape=shape_ord, kernel_initializer='VarianceScaling'))
model.add(MaxPooling2D(pool_size=(3, 3), padding='valid'))
model.add(Conv2D(32, kernel_size=(4, 4), padding='valid', activation='elu', strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(3, 3), padding='valid'))
model.add(Dropout(0.25))
model.add(Dense(1000, activation='elu'))
model.add(Dense(250, activation='elu'))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='elu'))
model.add(Dense(4, activation='softmax'))
model.summary()

plot_model(model, show_shapes=True, to_file='network.pdf')
