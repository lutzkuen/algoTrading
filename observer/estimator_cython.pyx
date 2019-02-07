import datetime
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
#import catboost as cb
import lightgbm as lgb
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
        #else:
        #    self.estimator = cb.CatBoostRegressor(loss_function='MAE')
        #    self.iscla = False
        #if '_close' in name:
        #    self.iscla = True
        #    if not estimpath:
        #        self.estimator = cb.CatBoostClassifier(iterations=1000)
        #else:
        #    self.iscla = False
        #    if not estimpath:

    def fit(self, x, y):
        return self.estimator.fit(x, y)

    def predict(self, x):
        if self.iscla:
            y_proba = self.estimator.predict_proba(x.reshape(1, -1))
            yp = [y_proba[0][1] - y_proba[0][0]][0]
            yp = max(min(1.0,yp),-1.0)
        else:
            yp = self.estimator.predict(x.reshape(1, -1))[0]
        return yp

    def get_feature_importances(self):
        return self.estimator.feature_importances_

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def improve_estimator(self, df, opt_table, estimtable=None, num_samples=1, estimpath=None, verbose = 1):
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

        params = {
                  'boosting_type': 'gbdt',
                  'objective': 'regression',
                  'metric': 'mae',
                  'max_depth': 10, 
                  'num_leaves': 60,
                  'learning_rate': 0.006,
                  'verbose': 0, 
                  'min_data_in_leaf': 4
                  #'early_stopping_round': 20
                  }
        n_estimators = 300

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.10, random_state=i)
        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)
        watchlist = [d_valid]

        self.estimator = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=100)
        ypred = self.estimator.predict(x)
        mae = np.mean(np.abs(ypred-y))
        print(self.name + ' -> ' + str(mae))
        if estimpath:
            self.save_estimator(estimpath)
        # now save the function evaluations to disk for later use
        estimator_score = {'name': self.name, 'score': mae}
        if estimtable:
            estimtable.upsert(estimator_score, ['name'])
        return mae

    def save_estimator(self, estim_path):
        estimator_name = estim_path + self.name
        pickle.dump(self.estimator, open(estimator_name, 'wb'))
