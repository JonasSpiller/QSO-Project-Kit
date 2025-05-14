## LIBRARIES
import numpy as np
import gc

## ML 
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, Flatten, LeakyReLU
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from utility_functions import ks, bs, nf


# CNN1t_1ConvA
def CNN_model1(hyperparameters, x_train, y_train, x_valid_test, y_valid_test, strategy):
    
    # =============================================================================
    # # Change this part for every new CNN
    # =============================================================================

    #batch_size, epochs, learning_rate, momentum, kernelsize, maxpooling = hyperparameters
    batch_size, epochs = hyperparameters
    learning_rate, momentum, kernelsize, maxpooling = 0.01, 0.5, 6, 3

    # =============================================================================
    # =============================================================================

    epochs = round(epochs)

    batch_size = bs(batch_size)
    kernelsize = ks(kernelsize)

    verbose = 1
    #decay_rate = learning_rate / epochs

    # x_train.shape = (nQSOs, wv_bins=7781, channels=1)
    # y_train.shape = (nQSOs, LAEclassification=1)
    n_bins, n_features = x_train.shape[1], x_train.shape[2]

    with strategy.scope():
        # =============================================================================
        # # Change this part for every new CNN
        # =============================================================================
        
        model_CNN1 = Sequential()
        model_CNN1.add(Conv1D(filters=batch_size, kernel_size=kernelsize, input_shape=(n_bins, n_features)))
        model_CNN1.add(MaxPooling1D(pool_size=round(maxpooling)))
        model_CNN1.add(Flatten())
        model_CNN1.add(Dense(1, activation='sigmoid'))
    
        # =============================================================================
        # =============================================================================
    
        lr_schedule = ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.96,  # here decay_rate becomes a factor (e.g., 0.96)
            staircase=True)
    
        #sgd = SGD(learning_rate=lr_schedule, momentum=momentum, nesterov=True)
        adam = Adam(learning_rate=lr_schedule)
        #model_CNN1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model_CNN1.compile(
            optimizer=adam,
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='pr_auc', curve='PR')  # Precision-Recall AUC
            ]
        )

        model_CNN1.summary()
        print("batch: " + str(batch_size) + " epochs: " + str(epochs) + " LR: " + str(learning_rate) + " momentum: " + str(
            momentum) + " kernel: " + str(kernelsize) + " maxpool: " + str(maxpooling))
    
        # fit network
        model_CNN1.fit(x_train, y_train, validation_data=(x_valid_test, y_valid_test), epochs=epochs, batch_size=batch_size, verbose=verbose)

    # results
    prediction_probsCNN1 = model_CNN1.predict(x_valid_test, batch_size=batch_size, verbose=1)
    y_test_hatCNN1 = np.argmax(prediction_probsCNN1, axis=1)

    # weights
    weights_CNN1 = model_CNN1.get_weights()

    # metrics
    scores_train = model_CNN1.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
    scores_valid_test = model_CNN1.evaluate(x_valid_test, y_valid_test, batch_size=batch_size, verbose=0)

    del model_CNN1
    gc.collect()

    return {'Y_pred': y_test_hatCNN1, 'Y_pred_prob': prediction_probsCNN1, 'accuracy_valid_test': scores_valid_test[1],
            'loss_valid_test': scores_valid_test[0], 'accuracy_train': scores_train[1], 'loss_train': scores_train[0],
            'weights': weights_CNN1}