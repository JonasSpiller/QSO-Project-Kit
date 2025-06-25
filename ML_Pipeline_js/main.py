## LIBRARIES
import pandas as pd
import numpy as np
import os
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import pickle
import time
import gc
import sys
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from utility_functions import extract_data_from_fits
from utility_functions import train_CNN_js
from MLmodels import *
from sklearn.model_selection import train_test_split
import datetime
import pytz

np.random.seed(100)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices: ", tf.config.list_physical_devices('GPU'))

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# For saving the progress to txt file
zurich_timezone = pytz.timezone('Europe/Zurich')

start_global = time.time()


data_path = os.path.expandvars('$SCRATCH/DATA/modifiedMainQSO/modifiedMainQSO_minsignal2/A/')
results_path = os.path.expandvars('$SCRATCH/RESULTS/modifiedMainQSO_minsignal2/A/')
os.makedirs(results_path, exist_ok=True)

flux_list = []
labels_list = []

fits_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.fits')]

for fits_file in fits_files:
    print(f"Processing file: {fits_file}")
    flux, labels, _ = extract_data_from_fits(fits_file)
    flux_list.append(flux)
    labels_list.append(labels)

# Concatenate all flux and labels arrays
all_flux = np.concatenate(flux_list, axis=0)
all_labels = np.concatenate(labels_list, axis=0)
print(len(all_labels))

x_train, x_testvalid, y_train, y_testvalid = train_test_split(all_flux, all_labels, train_size=0.7, random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_testvalid, y_testvalid, test_size=0.5, random_state=42)

# Standardize and then normalize each spectrum individually
x_train = np.array([(x - np.mean(x)) / np.std(x) for x in x_train])
x_valid = np.array([(x - np.mean(x)) / np.std(x) for x in x_valid])
x_test = np.array([(x - np.mean(x)) / np.std(x) for x in x_test])

# Normalize each standardized spectrum (min-max normalization to [-1, 1])
x_train = np.array([x / np.max(np.abs(x)) for x in x_train])
x_valid = np.array([x / np.max(np.abs(x)) for x in x_valid])
x_test = np.array([x / np.max(np.abs(x)) for x in x_test])

# Expand Dimensions to Channels
x_train = np.expand_dims(x_train, axis=-1)
x_valid = np.expand_dims(x_valid, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test  = y_test.astype('float32')

del all_flux, all_labels
gc.collect()



# =============================================================================
# # CNN
# =============================================================================

model = CNN_model5
# batch_size, epochs, learning_rate, momentum, kernelsize, maxpooling, (nfilters)
# bounds = [[16, 256], [10, 50]] #0
# bounds = [[16, 64], [100, 200], [0.0001, 0.01], [0.1, 0.9], [2, 8], [2, 3]] #1
# bounds = [[32, 128], [30, 80], [0.001, 0.01], [0.1, 0.9], [2, 8], [2, 3], [4, 64]] #2
# bounds = [[32, 128], [10, 100], [0.001, 0.01], [0.1, 0.9], [2, 8], [4, 128], [0.1, 0.8]] #3
#bounds = [[32, 128], [30, 80], [0.001, 0.01], [0.1, 0.9], [2, 8], [4, 64], [1e-6, 1e-1]] #4
bounds = [[32, 128], [10, 100], [0.001, 0.01], [0.1, 0.9], [2, 8], [4, 128], [2, 5]] #5

print("Starting")
res_opt_method, CM_opt, hyperparam_optim, optim_results = train_CNN_js(model, bounds, x_train, y_train, x_valid, y_valid, x_test, y_test, strategy)
print("Completed")


# =============================================================================
# # SAVE RESULTS
# =============================================================================

# use a dictionary with keys using the right names.
results = {'results': res_opt_method, 'confusion matrix': CM_opt, 'hyperparameters': hyperparam_optim,
           'y_test': list(y_test)}
a_file = open(results_path + "results5.pkl", "wb")
pickle.dump(results, a_file)
a_file.close()

# export optimization results
a_file = open(results_path + "GA_results5.pkl", "wb")
pickle.dump(optim_results, a_file)
a_file.close()

## OUT
end_global = time.time()
print(f"FINISHED, Runtime: {(end_global - start_global) / 60:.2f} minutes")
print(f"{hyperparam_optim=}")
