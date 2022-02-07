import os
import random
import sys


import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU, PReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error




sys.path.append("../src")
from utils.experiment import Bunch, make_experiment, make_experiment_tempfile

def build_model():
    model = Sequential()
    model.add(Input(shape=(8,)))
    model.add(Dense(544, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(672, kernel_initializer='normal', activation='relu'))
    model.add(Dense(960, kernel_initializer='normal', activation='relu'))
    model.add(Dense(736, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(192, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(160, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(160, kernel_initializer='normal', activation='relu'))

    #model.add(Dense(224, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(160, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(160, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='linear'))
    #model.add(PReLU())

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
            split='tuning for number of layers for fractal_dimension',
            split_lower=2.1,
            split_upper=2.5,
            max_layers=10,
            patience=10,
            epochs=500,
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
        X = df.iloc[:, :8]
        Y = df.iloc[:, 25:28]

        # Split on fractal dimension
        train_set = df[(df['fractal_dimension'] < params.split_lower) | (df['fractal_dimension'] > params.split_upper)]
        test_set = df[(df['fractal_dimension'] <= params.split_lower) & (df['fractal_dimension'] <= params.split_upper)]

        Y_train = train_set.iloc[:, 25:28]
        X_train = train_set.iloc[:, :8]
        Y_test = test_set.iloc[:, 25:28]
        X_test = test_set.iloc[:, :8]

        # Standardizing data and targets
        scaling_x = StandardScaler()
        scaling_y = StandardScaler()
        X_train = scaling_x.fit_transform(X_train)
        X_test = scaling_x.transform(X_test)
        Y_train = scaling_y.fit_transform(Y_train)
        Y_test = scaling_y.transform(Y_test)

        #Build NN model

        #model = build_model()#params.actuvation

        for i in range(0, params.max_layers):
            model = Sequential()
            model.add(Input(shape=(8,)))
            for j in range(0, i):
                model.add(Dense(512, kernel_initializer='normal', activation='relu'))

            model.add(Dense(3, kernel_initializer='normal', activation='linear'))
            #model.add(PReLU())

            # Compile model
            model.compile(loss=params.loss, optimizer='adam',
                          metrics=['mean_absolute_error'])

            print(model.summary())

            #Running and logging model plus Early stopping

            filepath = f"fractal_dimension_{_run._id}/best_model.hdf5"
            with make_experiment_tempfile('best_model.hdf5', _run, mode='wb', suffix='.hdf5') as model_file:
                #print(model_file.name)
                checkpoint = ModelCheckpoint(model_file.name, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

                # # patient early stopping
                es = EarlyStopping(monitor='val_loss', patience=params.patience, verbose=1)

                #log_csv = CSVLogger('fractal_dimension_loss_logs.csv', separator=',', append=False)

                callback_list = [checkpoint, es]
                history = model.fit(X_train, Y_train, epochs=params.epochs, batch_size=params.batch_size, validation_split=0.2, callbacks=callback_list)

                # choose the best Weights for prediction


                #Use best model to predict
                weights_file = f'fractal_dimension_{_run._id}/best_model.hdf5'  # choose the best checkpoint
                model.load_weights(model_file.name)  # load it
                model.compile(loss=params.loss, optimizer='adam', metrics=[params.loss])
            # Evaluate plus inverse transforms

            Y_test = scaling_y.inverse_transform(Y_test)
            Y_pred = model.predict(X_test)
            Y_pred = scaling_y.inverse_transform(Y_pred)

            # logging Y_test values
            Y_test = pd.DataFrame(data=Y_test, columns=["q_abs", "q_sca", "g"])


            error = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')
            tf.keras.backend.clear_session()



            print('Mean absolute error on test set [q_abs, q_sca, g]:-  ', error)


