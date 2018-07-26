import h5py

def save_all(hist, X_val, Y_val, w_val, string):
    hdf5_file = h5py.File('history_{}.h5'.format(string), mode='w')

    hdf5_file.create_dataset("val_loss", data=hist.history['val_loss'])
    hdf5_file.create_dataset("loss", data=hist.history['loss'])
    hdf5_file.create_dataset("val_accuracy", data=hist.history['val_acc'])
    hdf5_file.create_dataset("accuracy", data=hist.history['acc'])
    hdf5_file.close()

    hdf5_file = h5py.File('val_weights_{}.h5'.format(string), mode='w')

    hdf5_file.create_dataset("X_val", data=X_val)
    hdf5_file.create_dataset("Y_val", data=Y_val)
    hdf5_file.create_dataset("w_val", data=w_val)
    hdf5_file.close()
