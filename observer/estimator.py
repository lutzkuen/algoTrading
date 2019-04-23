import datetime
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import code
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


class Estimator(object):

    def __init__(self, name, estimpath=None):
        self.name = name
        self.library = 'catboost'
        self.iscla = False
        if estimpath:
            estimator_path = estimpath + name
            self.estimator = pickle.load(open(estimator_path, 'rb'))
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
        return self.estimator.predict(x.reshape(1, -1))[0]

    def get_feature_importances(self):
        return self.estimator.feature_importances_

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def improve_estimator(self, df, estimtable=None, num_samples=1, estimpath=None, verbose=1):
        x = np.array(df.values[:])
        y = np.array(df[self.name].values[:])  # make a deep copy to prevent data loss in future iterations
        y = y[num_samples:]  # drop first line
        x = x[:-num_samples, :]  # drop the last line
        print(y.shape)
        i = 0
        while i < y.shape[0]:
            if y[i] < -999990 or np.isnan(y[i]):  # missing values are marked with -999999
                y = np.delete(y, i)
                x = np.delete(x, i, axis=0)
            else:
                i += 1
        idx = [i for i in range(int(y.shape[0]/num_samples))]

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mse',
            'max_depth': 10,
            'num_leaves': 90,
            'learning_rate': 0.01,
            'verbose': 0,
            #'max_bin': 10000,
            'bagging_fraction': 0.5,
            'bagging_freq': 10,
            'min_data_in_leaf': 2
            # 'early_stopping_round': 20
        }
        n_estimators = 200

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.20, random_state=i)
        #x_train_idx, x_valid_idx, y_train_idx, y_valid_idx = train_test_split(idx, idx, test_size=0.15, random_state=i)
        #x_train = np.zeros((len(x_train_idx)*num_samples, x.shape[1]))
        #x_valid = np.zeros((len(x_valid_idx)*num_samples, x.shape[1]))
        #y_train = np.zeros((len(y_train_idx)*num_samples, ))
        #y_valid = np.zeros((len(y_valid_idx)*num_samples, ))
        #train_idx = 0
        #for i in x_train_idx:
        #    for j in range(num_samples):
        #        x_train[train_idx, :] = x[i*num_samples+j, :]
        #        y_train[train_idx] = y[i*num_samples+j]
        #        train_idx += 1
        #valid_idx = 0
        #for i in x_valid_idx:
        #    for j in range(num_samples):
        #        x_valid[valid_idx, :] = x[i*num_samples+j]
        #        y_valid[valid_idx] = y[i+num_samples+j]
        #        valid_idx += 1
        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)
        watchlist = [d_valid]

        self.estimator = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=100, early_stopping_rounds=10)
        ypred = self.estimator.predict(x_valid)
        mse = np.sqrt(np.mean((ypred - y_valid)**2))
        print(self.name + ' -> ' + str(mse))
        if estimpath:
            self.save_estimator(estimpath)
        # now save the function evaluations to disk for later use
        estimator_score = {'name': self.name, 'score': mse}
        if estimtable:
            estimtable.upsert(estimator_score, ['name'])
        return mse

    def save_estimator(self, estim_path):
        estimator_name = estim_path + self.name
        pickle.dump(self.estimator, open(estimator_name, 'wb'))
