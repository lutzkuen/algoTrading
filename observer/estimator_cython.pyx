import datetime
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


class Estimator(object):

    def __init__(self, name, estimpath=None):
        self.name = name
        self.library = 'sklearn'
        if estimpath:
            estimator_path = estimpath + name
            self.estimator = pickle.load(open(estimator_path, 'rb'))
        if '_close' in name:
            self.iscla = True
            if not estimpath:
                self.estimator = GradientBoostingClassifier(n_estimators=500)
        else:
            self.iscla = False
            if not estimpath:
                self.estimator = GradientBoostingRegressor(n_estimators=500)

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

    def improve_estimator(self, df, opt_table, estimtable=None, num_samples=1, estimpath=None):
        x = np.array(df.values[:])
        y = np.array(df[self.name].values[:])  # make a deep copy to prevent data loss in future iterations
        space = [Integer(1, 8, name='max_depth'),
                 Real(0.0001, 1, "log-uniform", name='learning_rate'),
                 Real(0.5, 1, "log-uniform", name='subsample'),
                 Integer(1, x.shape[1], name='max_features'),
                 Integer(2, 100, name='min_samples_split'),
                 Integer(1, 100, name='min_samples_leaf')]
        y = y[num_samples:]  # drop first line
        x = x[:-num_samples, :]  # drop the last line
        i = 0
        while i < y.shape[0]:
            if y[i] < -999990 or np.isnan(y[i]):  # missing values are marked with -999999
                y = np.delete(y, i)
                x = np.delete(x, i, axis=0)
            else:
                i += 1

        @use_named_args(space)
        def improve_objective(**params):
            print(params)
            self.estimator.set_params(**params)

            return -np.mean(cross_val_score(self.estimator, x, y, cv=3, n_jobs=1,
                                            scoring="neg_mean_absolute_error"))

        x0 = []
        y0 = []
        for opt_result in opt_table.find(colname=self.name, library=self.library):
            xs = [int(opt_result['max_depth']),
                  float(opt_result['learning_rate']),
                  float(opt_result['subsample']),
                  int(opt_result['max_features']),
                  int(opt_result['min_samples_split']),
                  int(opt_result['min_samples_leaf'])]
            ys = float(opt_result['function_value'])
            x0.append(xs)
            y0.append(ys)
        if len(y0) > 0:
            print('Using ' + str(len(y0)) + ' data points from previous runs')
            res_gp = gp_minimize(improve_objective, space, n_calls=10, n_random_starts=5, verbose=True, x0=x0, y0=y0)
        else:
            res_gp = gp_minimize(improve_objective, space, n_calls=10, n_random_starts=5, verbose=True)
        print("Best score=%.4f" % res_gp.fun)
        print("""Best parameters:
        - max_depth=%d
        - learning_rate=%.6f
        - subsample=%.6f
        - max_features=%d
        - min_samples_split=%d
        - min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2],
                                    res_gp.x[3], res_gp.x[4],
                                    res_gp.x[5]))
        self.estimator.set_params(n_estimators=500,
                                  max_depth=res_gp.x[0],
                                  learning_rate=res_gp.x[1],
                                  subsample=res_gp.x[2],
                                  max_features=res_gp.x[3],
                                  min_samples_split=res_gp.x[4],
                                  min_samples_leaf=res_gp.x[5])
        self.estimator.fit(x, y)  # , sample_weight=weights)
        if estimpath:
            self.save_estimator(estimpath)
        # now save the function evaluations to disk for later use
        estimator_score = {'name': self.name, 'score': -abs(res_gp.fun)}
        if estimtable:
            estimtable.upsert(estimator_score, ['name'])
        for xs, ys in zip(res_gp.x_iters, res_gp.func_vals):
            opt_result = {'max_depth': str(xs[0]),
                          'learning_rate': str(xs[1]),
                          'subsample': str(xs[2]),
                          'max_features': str(xs[3]),
                          'min_samples_split': str(xs[4]),
                          'min_samples_leaf': str(xs[5]),
                          'function_value': str(ys),
                          'colname': str(self.name),
                          'library': 'sklearn',
                          'timestamp': datetime.datetime.now()}
            opt_table.upsert(opt_result,
                             ['max_depth', 'learning_rate', 'subsample', 'max_features', 'min_samples_split',
                              'min_samples_leaf', 'colname', 'library'])
        return res_gp.fun

    def save_estimator(self, estim_path):
        estimator_name = estim_path + self.name
        pickle.dump(self.estimator, open(estimator_name, 'wb'))
