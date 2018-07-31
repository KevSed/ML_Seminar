import h5py
from performance import *

# With this function you get the output for a test file using the best trained DNN model


for i in [0,1,2,3]:

    model_evaluator(6,'test_sample.hdf5', 'FinalModel/', 'FinalModel/', i, unblind=True)
