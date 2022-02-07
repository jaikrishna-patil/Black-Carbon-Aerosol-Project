import os
import random
import sys


import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error




sys.path.append("../src")
from utils.experiment import Bunch, make_experiment, make_experiment_tempfile

def build_model():
    model = Sequential()
    model.add(Input(shape=(7,)))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(224, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(224, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(160, kernel_initializer='normal', activation='relu'))

    #model.add(Dense(224, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(160, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(160, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='linear'))

    # output_dense[:,0]=tf.keras.activations.sigmoid(output_dense[:,0])
    # output_q_abs=  tf.keras.layers.Activation(tf.nn.softplus)(output_dense[:,0:1])
    # output_q_sca= tf.keras.layers.Activation(tf.nn.softplus)(output_dense[:,1:2])
    # output_g= tf.keras.layers.Activation(tf.nn.sigmoid)(output_dense[:,2:3])
    # print(output_dense.shape)

    # model=tf.keras.Model(inputs=input_layer, outputs= [output_q_abs, output_q_sca, output_g])
    # model = tf.keras.Model(inputs=input_layer, outputs=output_dense)

    return model


if __name__ == '__main__':
    experiment = make_experiment()


    @experiment.config
    def config():
        params = dict(
            split='inverse_random_split',
            test_values=[],
            epochs=1000,
            batch_size=32,
            #n_hidden=8,
            #dense_units=[416, 288, 256,256, 192,448,288,128, 352,224],
            #kernel_initializer=['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal'],
            activation='relu',
            loss='mean_squared_error'
            #range=(10, 15)
        )


    @experiment.automain
    def main(params, _run):

        params = Bunch(params)

        #Load dataset
        df = pd.read_excel('database_new.xlsx')
        #X = df.iloc[:, [0, 3, 6, 24, 25, 26, 27]]
        X = df.iloc[:, [0, 24, 25, 26, 27]]
        Y = df.iloc[:, [1, 2]]

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y,
            test_size=0.25,
            random_state=42)

        # Standardizing data and targets
        scaling_x = StandardScaler()
        scaling_y = StandardScaler()
        X_train = scaling_x.fit_transform(X_train)
        X_test = scaling_x.transform(X_test)
        Y_train = scaling_y.fit_transform(Y_train)
        Y_test = scaling_y.transform(Y_test)

        #Build NN model

        model = build_model()#params.actuvation

        #Compile model
        model.compile(loss=params.loss, optimizer='adam',
                      metrics=['mean_absolute_error'])

        print(model.summary())


        #Running and logging model plus Early stopping

        filepath = f"inverse_random_split_{_run._id}/best_model.hdf5"
        with make_experiment_tempfile('best_model.hdf5', _run, mode='wb', suffix='.hdf5') as model_file:
            #print(model_file.name)
            checkpoint = ModelCheckpoint(model_file.name, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

            # # patient early stopping
            es = EarlyStopping(monitor='val_loss', patience=200, verbose=1)

            #log_csv = CSVLogger('fractal_dimension_loss_logs.csv', separator=',', append=False)

            callback_list = [checkpoint, es]
            history = model.fit(X_train, Y_train, epochs=params.epochs, batch_size=params.batch_size, validation_split=0.2, callbacks=callback_list)

            # choose the best Weights for prediction

            #Save the model

            #Save metrics loss and val_loss
            #print(history.history.keys())
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = len(loss)
            print(epochs)
            for epoch in range(0, epochs):

                # Log scalar wil log a single number. The string is the metrics name
                _run.log_scalar('Training loss', loss[epoch])
                _run.log_scalar('Validation loss', val_loss[epoch])

            #Use best model to predict
            weights_file = f'inverse_random_split_{_run._id}/best_model.hdf5'  # choose the best checkpoint
            model.load_weights(model_file.name)  # load it
            model.compile(loss=params.loss, optimizer='adam', metrics=[params.loss])
        #Evaluate plus inverse transforms

        Y_test = scaling_y.inverse_transform(Y_test)
        Y_pred = model.predict(X_test)
        Y_pred = scaling_y.inverse_transform(Y_pred)

        #logging Y_test values
        Y_test = pd.DataFrame(data=Y_test, columns=["fractal_dimension", "fraction_of_coating"])
        #Y_test.reset_index(inplace=True, drop=True)
        for i in Y_test['fractal_dimension']:
            _run.log_scalar('Actual fractal_dimension', i)
        for i in Y_test['fraction_of_coating']:
            _run.log_scalar('Actual fraction_of_coating', i)

        #logging predicted values
        Y_pred = pd.DataFrame(data=Y_pred, columns=["fractal_dimension", "fraction_of_coating"])
        for i in Y_pred['fractal_dimension']:
            _run.log_scalar('Predicted fractal_dimension', i)
        for i in Y_pred['fraction_of_coating']:
            _run.log_scalar('Predicted fraction_of_coating', i)

        # logging difference between the two
        Y_diff = Y_test - Y_pred
        for i in Y_diff['fractal_dimension']:
            _run.log_scalar('Absolute error fractal_dimension', i)
        for i in Y_diff['fraction_of_coating']:
            _run.log_scalar('Absolute error fraction_of_coating', i)

        error = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')



        #error=error*100
        print('Mean absolute error on test set [fractal_dimension, fraction_of_coating]:-  ', error)



