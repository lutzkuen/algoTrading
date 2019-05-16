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

# KNOCKOUT_COLUMNS = ['EUR_AUD_open','USB10Y_USD_close','XAU_EUR_ma30','USD_JPY_vol','USD_SAR_low','AUD_JPY_ma30','BCO_USD_open','GBP_CHF_open','USD_MXN_ma30','EUR_CHF_vol','GBP_SGD_ma30','GBP_PLN_vol','USD_NOK_high','TWIX_USD_open','USD_HUF_vol','XAG_NZD_high','USB10Y_USD_open','GBP_SGD_open','NZD_CAD_ma30','UK100_GBP_vol','AUD_USD_high','USD_HKD_vol','XAU_HKD_vol','XPT_USD_vol','NL25_EUR_ma30','XAU_XAG_low','CHF_JPY_ma30','XAG_SGD_vol','CHF_ZAR_vol','AUD_HKD_open','AUD_NZD_open','XAG_USD_high','EUR_ZAR_open','USB05Y_USD_open','TRY_JPY_vol','NZD_CAD_open','USD_CZK_low','XAG_AUD_vol','XAU_USD_close','USD_THB_vol','HKD_JPY_ma30','EUR_SGD_vol','USD_SGD_vol','CAD_SGD_ma30','XAU_SGD_vol','SUGAR_USD_vol','USD_SAR_high','HKD_JPY_vol','USD_CZK_vol','EUR_NZD_vol','EUR_CZK_ma30','USD_INR_open','USD_DKK_low','AUD_SGD_vol','XAG_USD_low','NZD_USD_high','XAU_USD_high','GBP_JPY_ma30','XAG_EUR_low','XAU_CHF_ma30','AUD_USD_vol','XAG_HKD_open','XAG_HKD_high','GBP_CHF_ma30','XAG_HKD_low','EUR_JPY_open','NZD_USD_vol','HK33_HKD_vol','USD_CHF_ma30','CN50_USD_vol','XAG_JPY_high','GBP_USD_low','AUD_USD_open','USB30Y_USD_low','USD_CNH_close','EUR_SGD_high','WTICO_USD_open','XAU_AUD_ma30','JP225_USD_open','USB02Y_USD_low','GBP_CHF_vol','USD_CNH_high','CAD_SGD_vol','CN50_USD_open','CHF_ZAR_open','AUD_HKD_ma30','AUD_CAD_vol','XAU_GBP_open','NATGAS_USD_open','EUR_AUD_ma30','XAU_CHF_open','CAD_CHF_vol','GBP_NZD_vol','EUR_USD_open','AUD_NZD_ma30','USD_ZAR_vol','USD_JPY_close','NL25_EUR_vol','XAU_CAD_open','USD_CAD_open','USD_JPY_ma30','XAG_CAD_open','AU200_AUD_vol','USD_CNH_low','EUR_SEK_vol','SGD_JPY_vol','XAU_AUD_open','XAG_CHF_high','USD_NOK_ma30','USD_SAR_open','AU200_AUD_open','XPT_USD_open','EUR_JPY_ma30','XAU_USD_vol','XAG_SGD_high','SGD_CHF_vol','GBP_SGD_vol','ZAR_JPY_open','EUR_USD_vol','GBP_AUD_ma30','USD_CHF_vol','CAD_HKD_open','USD_CNH_vol','GBP_ZAR_open','XAU_HKD_open','NZD_HKD_open','EUR_SGD_open','USB05Y_USD_ma30','EUR_SEK_open','TWIX_USD_vol','GBP_NZD_open','XAU_GBP_ma30','GBP_AUD_vol','SGD_HKD_open','EU50_EUR_open','XAU_HKD_high','EURLowImpactExpected_next','USD_SGD_open','IN50_USD_vol','XCU_USD_open','XAU_EUR_vol','CAD_HKD_vol','GBPHighImpactExpected_next2','XAU_XAG_vol','NZD_HKD_vol','SG30_SGD_high','USD_DKK_vol','CHF_ZAR_ma30','EUR_PLN_ma30','EUR_ZAR_ma30','AUD_SGD_ma30','US2000_USD_open','WTICO_USD_ma30','CHF_HKD_vol','USD_ZAR_ma30','CHFLowImpactExpected_next2','EURMediumImpactExpected_next2','GBP_CAD_vol','HK33_HKD_open','XAU_NZD_ma30','XAU_HKD_ma30','EUR_CHF_open','USB02Y_USD_open','USB30Y_USD_open','XAU_AUD_vol','USD_DKK_open','NZD_CHF_vol','XAU_NZD_open','XAU_USD_open','GBP_CAD_open','USD_TRY_vol','SGD_HKD_vol','XAU_SGD_ma30','XAU_USD_ma30','ZAR_JPY_ma30','EUR_CAD_open','US2000_USD_ma30','XAG_AUD_open','EUR_HUF_open','NZD_USD_ma30','EUR_CHF_ma30','XAG_JPY_open','XAG_CHF_vol','XAU_JPY_vol','IN50_USD_open','EUR_NZD_open','EUR_sentiment_Low Impact Expected','AUD_NZD_vol','NZD_HKD_ma30','EURLowImpactExpected_next2','EUR_HKD_vol','XAG_USD_vol','XAG_AUD_ma30','XAU_XAG_ma30','XAG_CHF_open','USD_HUF_ma30','JPYLowImpactExpected','NAS100_USD_vol','EUR_SEK_ma30','EUR_DKK_open','GBPLowImpactExpected_next','XAG_NZD_vol','SG30_SGD_open','GBP_ZAR_ma30','USD_HUF_open','XAG_NZD_ma30','US30_USD_vol','NZD_USD_open','JPYLowImpactExpected_next','EUR_CAD_ma30','XAG_USD_open','USDLowImpactExpected_next2','GBP_HKD_vol','XAU_EUR_open','GBP_HKD_open','XAG_EUR_vol','GBP_AUD_open','GBP_JPY_open','USD_THB_ma30','EUR_sentiment_Medium Impact Expected_next2','XAG_CAD_vol','NZD_CHF_ma30','EUR_SGD_ma30','USB10Y_USD_ma30','GBP_NZD_ma30','USD_SEK_open','NZD_SGD_open','USD_sentiment_Low Impact Expected_next2','USD_CZK_open','CNYMediumImpactExpected_next2','AUDLowImpactExpected_next2','EUR_HUF_ma30','EUR_HKD_open','NZDLowImpactExpected_next2','JPY_sentiment_Low Impact Expected','US2000_USD_vol','US30_USD_open','XAU_CHF_vol','AUD_SGD_open','GBP_USD_open','USD_CNH_ma30','IN50_USD_ma30','XAU_GBP_vol','NZDLowImpactExpected_next','EUR_NZD_ma30','JP225_USD_ma30','USD_SEK_ma30','AUD_USD_ma30','USB30Y_USD_ma30','CNYLowImpactExpected_next2','NAS100_USD_ma30','GBP_USD_vol','XAU_CAD_vol','EUR_DKK_ma30','EURMediumImpactExpected_next','XAG_GBP_ma30','USD_PLN_open','XAG_NZD_open','EUR_HKD_ma30','USD_THB_open','CNY_sentiment_Medium Impact Expected_next2','USD_TRY_open','USD_CAD_vol','CADMediumImpactExpected_next2','XAG_CHF_ma30','XAG_USD_ma30','NZDMediumImpactExpected','EUR_sentiment_Low Impact Expected_next','NZD_SGD_ma30','NZD_SGD_vol','XAU_NZD_vol','XPD_USD_ma30','GBP_sentiment_Low Impact Expected','XAG_EUR_ma30','USD_CNH_open','AUDLowImpactExpected_next','XAG_HKD_ma30','US30_USD_ma30','XAG_GBP_vol','CADHighImpactExpected_next2','NZDHighImpactExpected_next','TRY_JPY_open','USDLowImpactExpected','USD_SGD_ma30','XAG_JPY_ma30','JPY_sentiment_Low Impact Expected_next2','USD_PLN_ma30','weekday','XAG_GBP_open','NZD_sentiment_Medium Impact Expected','JPYLowImpactExpected_next2','JPY_sentiment_Low Impact Expected_next','USDLowImpactExpected_next','CNYHighImpactExpected_next2','SGD_HKD_ma30','XAG_EUR_open','XPD_USD_open','GBPMediumImpactExpected_next','AUD_sentiment_Medium Impact Expected','CNY_sentiment_Medium Impact Expected_next','XAG_SGD_open','CHFLowImpactExpected_next','EUR_USD_ma30','EUR_TRY_open','AUDHighImpactExpected_next','XAG_CAD_ma30','SUGAR_USD_open','EUR_TRY_ma30','GBPLowImpactExpected','AUD_sentiment_High Impact Expected','CNY_sentiment_Medium Impact Expected','EUR_sentiment_Low Impact Expected_next2','GBP_sentiment_High Impact Expected_next2','AUDLowImpactExpected','NAS100_USD_open','USD_DKK_ma30','SPX500_USD_open','USDMediumImpactExpected_next2','USDMediumImpactExpected','AUDHighImpactExpected_next2','GBP_sentiment_Medium Impact Expected','JPY_sentiment_Medium Impact Expected','CADHighImpactExpected_next','GBP_sentiment_Low Impact Expected_next2','AUDHighImpactExpected','SPX500_USD_ma30','CNY_sentiment_Low Impact Expected_next2','GBP_sentiment_Low Impact Expected_next','USB02Y_USD_ma30','USD_TRY_ma30','EURHighImpactExpected_next2','CNYLowImpactExpected_next','TRY_JPY_ma30','GBP_sentiment_Medium Impact Expected_next','USD_sentiment_Medium Impact Expected_next2','NZDLowImpactExpected','USD_CZK_ma30','GBPMediumImpactExpected_next2','GBPMediumImpactExpected','GBP_sentiment_Medium Impact Expected_next2','GBPLowImpactExpected_next2','USD_sentiment_Low Impact Expected','JPYMediumImpactExpected_next','AUD_sentiment_Medium Impact Expected_next','CADLowImpactExpected','USDHighImpactExpected_next','AUDMediumImpactExpected','NZD_sentiment_Low Impact Expected','AUD_sentiment_Low Impact Expected','AUD_sentiment_High Impact Expected_next','XAG_SGD_ma30','XAU_XAG_open','AUD_sentiment_Low Impact Expected_next','EURMediumImpactExpected','CAD_sentiment_Low Impact Expected_next2','USDMediumImpactExpected_next','CNYMediumImpactExpected','NZD_sentiment_Medium Impact Expected_next','USD_sentiment_Medium Impact Expected','USD_sentiment_High Impact Expected','CHFLowImpactExpected','NZD_sentiment_High Impact Expected','CNY_sentiment_Low Impact Expected','USDHighImpactExpected','NZD_sentiment_Low Impact Expected_next','CHF_sentiment_Low Impact Expected','JPY_sentiment_Medium Impact Expected_next2','JPY_sentiment_Medium Impact Expected_next','GBP_sentiment_High Impact Expected','USD_sentiment_High Impact Expected_next2','NZDHighImpactExpected','USD_sentiment_Low Impact Expected_next','CNYMediumImpactExpected_next','CAD_sentiment_Medium Impact Expected','CHF_sentiment_Low Impact Expected_next2','JPYMediumImpactExpected','JPYHighImpactExpected_next2','AUDMediumImpactExpected_next','GBPHighImpactExpected_next','USD_sentiment_Medium Impact Expected_next','CHFHighImpactExpected_next','CADLowImpactExpected_next2','CNY_sentiment_Low Impact Expected_next','NZDHighImpactExpected_next2','EUR_sentiment_Medium Impact Expected_next','CADLowImpactExpected_next','EUR_sentiment_Medium Impact Expected','EURHighImpactExpected_next','AUD_sentiment_Medium Impact Expected_next2','CAD_sentiment_Low Impact Expected','CNYLowImpactExpected','EURHighImpactExpected','JPYHighImpactExpected','EURNon-Economic_next2','CNYHighImpactExpected','CHFMediumImpactExpected_next2','CADMediumImpactExpected_next','NZD_sentiment_High Impact Expected_next','CNYNon-Economic_next','CAD_sentiment_Low Impact Expected_next','AUDNon-Economic_next','CADHighImpactExpected','GBP_sentiment_High Impact Expected_next','GBPHighImpactExpected','USD_sentiment_High Impact Expected_next','AUDMediumImpactExpected_next2','AUD_sentiment_Low Impact Expected_next2','CAD_sentiment_Medium Impact Expected_next','CHF_sentiment_Medium Impact Expected_next','CADMediumImpactExpected','CNY_sentiment_High Impact Expected','USDNon-Economic','CHFMediumImpactExpected_next','JPYHighImpactExpected_next','NZDMediumImpactExpected_next2','AUD_sentiment_High Impact Expected_next2','CNY_sentiment_High Impact Expected_next','NZD_sentiment_High Impact Expected_next2','CHF_sentiment_Medium Impact Expected','JPYNon-Economic','CHFMediumImpactExpected','CAD_sentiment_High Impact Expected_next2','NZDMediumImpactExpected_next','NZD_sentiment_Low Impact Expected_next2','CNYNon-Economic_next2','EURNon-Economic_next','CNY_sentiment_High Impact Expected_next2','GBPNon-Economic','CADNon-Economic','CNYNon-Economic','CNYHighImpactExpected_next','CHF_sentiment_Low Impact Expected_next','CAD_sentiment_Medium Impact Expected_next2','NZDNon-Economic_next','JPYMediumImpactExpected_next2','CHF_sentiment_Medium Impact Expected_next2','CHFHighImpactExpected_next2','CAD_sentiment_High Impact Expected','CAD_sentiment_High Impact Expected_next','NZD_sentiment_Medium Impact Expected_next2','JPYNon-Economic_next','JPY_sentiment_High Impact Expected_next2','USDNon-Economic_next2','CADNon-Economic_next','CADNon-Economic_next2','CHFNon-Economic','EURNon-Economic','AUDNon-Economic','EUR_sentiment_High Impact Expected_next','EUR_sentiment_High Impact Expected','JPYNon-Economic_next2','GBPNon-Economic_next','JPY_sentiment_High Impact Expected','CHFNon-Economic_next','CHF_sentiment_High Impact Expected','JPY_sentiment_High Impact Expected_next','EUR_sentiment_High Impact Expected_next2','CHFNon-Economic_next2','AUDNon-Economic_next2','NZDNon-Economic_next2','NZDNon-Economic','GBPNon-Economic_next2','USDNon-Economic_next','CHF_sentiment_High Impact Expected_next2','CHF_sentiment_High Impact Expected_next','AUD_sentiment_Non-Economic','AUD_sentiment_Non-Economic_next','AUD_sentiment_Non-Economic_next2','CAD_sentiment_Non-Economic','CAD_sentiment_Non-Economic_next','CAD_sentiment_Non-Economic_next2','CHFHighImpactExpected','CHF_sentiment_Non-Economic','CHF_sentiment_Non-Economic_next','CHF_sentiment_Non-Economic_next2','CNY_sentiment_Non-Economic','CNY_sentiment_Non-Economic_next','CNY_sentiment_Non-Economic_next2','EUR_sentiment_Non-Economic','EUR_sentiment_Non-Economic_next','EUR_sentiment_Non-Economic_next2','GBP_sentiment_Non-Economic','GBP_sentiment_Non-Economic_next','GBP_sentiment_Non-Economic_next2','JPY_sentiment_Non-Economic','JPY_sentiment_Non-Economic_next','JPY_sentiment_Non-Economic_next2','NZD_sentiment_Non-Economic','NZD_sentiment_Non-Economic_next','NZD_sentiment_Non-Economic_next2','USD_sentiment_Non-Economic','USD_sentiment_Non-Economic_next','USD_sentiment_Non-Economic_next2']


class Estimator(object):

    def __init__(self, name, estimpath=None):
        self.name = name
        self.library = 'catboost'
        self.iscla = False
        self.knockout = []
        if estimpath:
            estimator_path = estimpath + name
            knockout_path = estimpath + name + '.knockout'
            self.estimator = pickle.load(open(estimator_path, 'rb'))
            self.knockout = pickle.load(open(knockout_path, 'rb'))
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

    def clean_df(self, df):
        try:
            for col in df.columns:
                if '_open' in col:
                    df = df.drop(col, axis=1)
        except:
            for col in df.index:
                if '_open' in col:
                    df = df.drop(col)
        return df

    def predict(self, _x):
        x = _x.copy()
        x = self.clean_df(x)
        # code.interact(banner='', local=locals())
        for col in self.knockout:
            try:
                x = x.drop(col)
            except Exception as e:
                print('Could not drop ' + str(col))
                # code.interact(banner='', local=locals())
        return self.estimator.predict(np.array(x.values[:]).reshape(1, -1))[0]

    def get_feature_importances(self):
        return self.estimator.feature_importances_

    def set_params(self, **params):
        return self.estimator.set_params(**params)

    def improve_estimator(self, _df, estimtable=None, num_samples=1, estimpath=None, verbose=1):
        max_features = 25 # as a rule of thumb the number of features should not be greater than sqrt(num samples)
        i_knockout = 0
        knockout_columns = []
        df = _df.copy()
        df = self.clean_df(df)
        self.estimator = None
        is_finished = False
        best_mse = 999999
        best_mae = 999999
        while not is_finished or len(df.columns) > max_features:
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
                'metric': 'mae',
                'max_depth': 10,
                'num_leaves': 10,
                'learning_rate': 0.001,
                # 'feature_fraction': 0.2,
                'verbose': 0
                #'max_bin': 10000,
                # 'bagging_fraction': 0.5,
                # 'bagging_freq': 10,
                # 'min_data_in_leaf': 2
                # 'early_stopping_round': 20
            }
            n_estimators = 10000
            x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.30)
            d_train = lgb.Dataset(x_train, label=y_train)
            d_valid = lgb.Dataset(x_valid, label=y_valid)
            watchlist = [d_valid]
            estim = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=10000, early_stopping_rounds=100)
            importances = estim.feature_importance(importance_type='gain')
            importance_arr = []
            for label, importance in zip(df.columns, importances):
                importance_arr.append({'label': label, 'importance': importance})
            ypred = estim.predict(x_valid)
            mse = np.sqrt(np.mean((ypred - y_valid)**2))
            mae = np.mean(np.abs(ypred - y_valid))
            if mse < best_mse or len(df.columns) > max_features:
                i_knockout += len(knockout_columns)
                for cc in knockout_columns:
                    self.knockout.append(cc)
                knockout_columns = []
                self.estimator = estim
                best_mse = mse
                best_mae = mae
                # knock out all columns which contribute less than half to the gain
                mean_gain = np.quantile(importances, 0.1)
                current_importances = []
                for label, importance in zip(df.columns, importances):
                    current_importances.append({'label': label, 'importance': importance})
                    #if importance <= mean_gain and not label == self.name:
                    #    knockout_columns.append(label)
                current_importances = sorted(current_importances, key = lambda x: x.get('importance'), reverse=True)
                for i in range(int(len(current_importances)/2)):
                    if not current_importances[i].get('label') == self.name:
                        knockout_columns.append(current_importances[i].get('label'))
                df = df.drop(knockout_columns, axis=1)
                print('Remaining Features: ' + str(len(df.columns)))
            else:
                is_finished = True
        best_mse = best_mse / np.mean(abs(y))
        best_mae = best_mae / np.mean(abs(y))
        print(self.name + ' -> ' + str(best_mse) + ' / ' + str(best_mae) + ' (knocked out ' + str(i_knockout) + ' columns, ' + str(len(df.columns)) + ' features remaining )')
        if estimpath:
            self.save_estimator(estimpath)
        # now save the function evaluations to disk for later use
        estimator_score = {'name': self.name, 'score': best_mse}
        if estimtable:
            estimtable.upsert(estimator_score, ['name'])
        return importance_arr, best_mse, best_mae

    def save_estimator(self, estim_path):
        estimator_name = estim_path + self.name
        pickle.dump(self.estimator, open(estimator_name, 'wb'))
        knockout_path = estimator_name + '.knockout'
        pickle.dump(self.knockout, open(knockout_path, 'wb'))
