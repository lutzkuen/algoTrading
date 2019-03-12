import datetime
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from keras.activations import relu
from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape, MaxPooling3D
from keras.utils import np_utils
from keras import regularizers
import keras
import code
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


class Estimator(object):

    def __init__(self, name, input_size):
        self.name = name
        self.library = 'keras'
        self.weights_file = '/home/tubuntu/data/keras.h5'
        self.model = self.create_network(input_size)

        # else:
        #    self.estimator = cb.CatBoostRegressor(loss_function='MAE')
        #    self.iscla = False
        # if '_close' in name:
        #    self.iscla = True
        #    if not estimpath:
        #        self.estimator = cb.CatBoostClassifier(iterations=1000)
        # else:
        #    self.iscla = False
        #    if not estimpath:

    def fit(self, x, y):
        return self.estimator.fit(x, y)

    def predict(self, x):
        if self.iscla:
            y_proba = self.estimator.predict_proba(x.reshape(1, -1))
            yp = [y_proba[0][1] - y_proba[0][0]][0]
            yp = max(min(1.0, yp), -1.0)
        else:
            yp = self.estimator.predict(x.reshape(1, -1))[0]
        return yp

    def get_feature_importances(self):
        return self.estimator.feature_importances_

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def create_network(self, kernel_len, num_layers=4):
        initializer = keras.initializers.Identity()
        model = Sequential()
        for i in range(num_layers):
            model.add(Dense(kernel_len, activation='relu', kernel_initializer=initializer))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mean_squared_error'])
        try:
            model.load_weights(weights_file)
        except:
            print('could not load weights')
        return model

    def improve_estimator(self, df, opt_table, name, estimtable=None, num_samples=1, estimpath=None, verbose=1):
        x = np.array(df.values[:])
        y = x.copy()
        y = y[num_samples:]  # drop first line
        x = x[:-num_samples, :]  # drop the last line
        print(y.shape)
        i = 0
        while i < y.shape[0]:
            if np.any(y[i, :] < -999990) or np.any(np.isnan(y[i, :])):  # missing values are marked with -999999
                y = np.delete(y, i, axis=0)
                x = np.delete(x, i, axis=0)
            else:
                i += 1

        self.model.fit(x, y, validation_split=0.1, epochs=100)
        ypred = self.model.predict(x)
        mse = np.sqrt(np.mean((ypred[y.columns.get_loc(name) - y[name])**2))
        print(name + ' -> ' + str(mse))
        if estimpath:
            self.save_estimator(estimpath)
        # now save the function evaluations to disk for later use
        estimator_score = {'name': name, 'score': mse}
        #if estimtable:
        #    estimtable.upsert(estimator_score, ['name'])
        return mse

    def save_estimator(self, estim_path):
        self.model.save_weights(self.weights_file)
