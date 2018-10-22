#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Callable

__author__ = "Lutz Künneke"
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Lutz Künneke"
__email__ = "lutz.kuenneke89@gmail.com"
__status__ = "Prototype"
"""
Candle logger and ML controller
Use at own risk
Author: Lutz Kuenneke, 26.07.2018
"""
import json

try:
    # noinspection PyUnresolvedReferences
    import v20
    # noinspection PyUnresolvedReferences
    from v20.request import Request

    v20present = True
except ImportError:
    print('WARNING: V20 library not present. Connection to broker not possible')
    v20present = False

import code

import progressbar
import configparser
import math
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import dataset
import pickle
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline


def prev_working_day(_day):
    date1 = datetime.datetime.strptime(_day, '%Y-%m-%d')
    wd = date1.weekday()
    if wd < 6:
        date2 = date1 - datetime.timedelta(days=1)
    else:
        date2 = date1 - datetime.timedelta(days=3)  # skip to friday on weekends
    return date2.strftime('%Y-%m-%d')


def merge_dicts(dict1, dict2, suffix):
    for key in dict2.keys():
        dict1[key + suffix] = dict2[key]
    return dict1


def get_range_int(val, change, lower=-math.inf, upper=math.inf):
    lval = math.floor(val * (1 - change))
    uval = math.ceil(val * (1 + change))
    rang = []
    if lower <= lval < val:
        rang.append(lval)
    rang.append(val)
    if val < uval <= upper:
        rang.append(uval)
    return rang


def get_range_flo(val, change, lower=-math.inf, upper=math.inf):
    lval = val * (1 - change)
    uval = val * (1 + change)
    rang = []
    if lower <= lval < val:
        rang.append(lval)
    rang.append(val)
    if val < uval <= upper:
        rang.append(uval)
    return rang


def get_gb_importances(gb, x, y):
    gb.fit(x, y)
    return gb.feature_importances_


class EstimatorPipeline(object):
    feature_importances_: object

    def __init__(self, percentile=50, learning_rate=0.1, n_estimators=100, min_samples_split=2, path=None,
                 classifier=False):
        self.classifier = classifier
        if path:  # read from disk
            gb_path = path + '.gb'
            self.gb = pickle.load(open(gb_path, 'rb'))
            percentile_path = path + '.pipe'
            percentile_attr = pickle.load(open(percentile_path, 'rb'))
            param_path = path + '.param'
            self.params = pickle.load(open(param_path, 'rb'))
            score_func: Callable[[Any, Any], Any] = lambda x, y: get_gb_importances(self.gb, x, y)
            #score_func = lambda x, y: get_gb_importances(self.gb, x, y)
            self.percentile = SelectPercentile(score_func=score_func, percentile=self.params.get('percentile'))
            self.percentile.scores_ = percentile_attr.get('scores')
            self.percentile.pvalues_ = percentile_attr.get('pvalues')
            # code.interact(banner='', local=locals())
        else:
            self.params = {'percentile': percentile, 'learning_rate': learning_rate, 'n_estimators': n_estimators,
                           'min_samples_split': min_samples_split}
            if classifier:
                self.gb = GradientBoostingClassifier(learning_rate=learning_rate, min_samples_split=min_samples_split,
                                                     n_estimators=n_estimators)
            else:
                self.gb = GradientBoostingRegressor(learning_rate=learning_rate, min_samples_split=min_samples_split,
                                                    n_estimators=n_estimators)
            #score_func = lambda x, y: get_gb_importances(self.gb, x, y)
            score_func: Callable[[Any, Any], Any] = lambda x, y: get_gb_importances(self.gb, x, y)
            self.percentile = SelectPercentile(score_func=score_func, percentile=percentile)
        self.pipeline = make_pipeline(
            self.percentile,
            self.gb
        )

    def get_feature_importances(self):
        return self.gb.feature_importances_

    def get_params(self, deep = True):# keyword deep needed for gridsearch
        return self.params

    def fit(self, x, y, sample_weight=None):
        if self.classifier:
            self.pipeline.fit(x, y, gradientboostingclassifier__sample_weight=sample_weight)
        else:
            self.pipeline.fit(x, y, gradientboostingregressor__sample_weight=sample_weight)
        self.feature_importances_ = self.gb.feature_importances_
        return self

    def write_to_disk(self, path):
        gb_path = path + '.gb'
        pickle.dump(self.gb, open(gb_path, 'wb'))
        pipe_path = path + '.pipe'
        pipe_attr = {'scores': self.percentile.scores_, 'pvalues': self.percentile.pvalues_}
        pickle.dump(pipe_attr, open(pipe_path, 'wb'))
        param_path = path + '.param'
        pickle.dump(self.params, open(param_path, 'wb'))

    def predict(self, x):
        return self.pipeline.predict(x)

    def score(self, x, y=None, sample_weight=None):
        return self.pipeline.score(x, y=y, sample_weight=sample_weight)

    def set_params(self, percentile=None, learning_rate=None, n_estimators=None, min_samples_split=None):
        if percentile:
            self.percentile.set_params(percentile=percentile)
            self.params['percentile'] = percentile
        if learning_rate:
            self.gb.set_params(learning_rate=learning_rate)
            self.params['learning_rate'] = learning_rate
        if n_estimators:
            self.gb.set_params(n_estimators=n_estimators)
            self.params['n_estimators'] = n_estimators
        if min_samples_split:
            self.gb.set_params(min_samples_split=min_samples_split)
            self.params['min_samples_split'] = min_samples_split
        return self


class Controller(object):
    def __init__(self, config_name, _type, verbose=2):
        config = configparser.ConfigParser()
        config.read(config_name)
        self.verbose = verbose

        self.settings = {'estim_path': config.get('data', 'estim_path'),
                         'prices_path': config.get('data', 'prices_path')}
        if _type and v20present:
            self.settings['domain'] = config.get(_type,
                                                 'streaming_hostname')
            self.settings['access_token'] = config.get(_type, 'token')
            self.settings['account_id'] = config.get(_type,
                                                     'active_account')
            self.settings['v20_host'] = config.get(_type, 'hostname')
            self.settings['v20_port'] = config.get(_type, 'port')
            self.settings['account_risk'] = int(config.get('triangle', 'account_risk'))
            self.oanda = v20.Context(self.settings.get('v20_host'),
                                     port=self.settings.get('v20_port'),
                                     token=self.settings.get('access_token'
                                                             ))
            self.allowed_ins = \
                self.oanda.account.instruments(self.settings.get('account_id'
                                                                 )).get('instruments', '200')
            self.trades = self.oanda.trade.list_open(self.settings.get('account_id')).get('trades', '200')
        self.db = dataset.connect(config.get('data', 'candle_path'))
        self.calendar_db = dataset.connect(config.get('data', 'calendar_path'))
        self.calendar = self.calendar_db['calendar']
        self.table = self.db['dailycandles']
        self.estimtable = self.db['estimators']
        self.importances = self.db['feature_importances']

    def retrieve_data(self, num_candles, completed=True, upsert=False):
        for ins in self.allowed_ins:
            candles = self.get_candles(ins.name, 'D', num_candles)
            self.candles_to_db(candles, ins.name, completed=completed, upsert=upsert)

    def get_pip_size(self, ins):
        pip_loc = [_ins.pipLocation for _ins in self.allowed_ins if _ins.name == ins]
        if not len(pip_loc) == 1:
            return None
        return -pip_loc[0] + 1

    def get_spread(self, ins):
        args = {'instruments': ins}
        price_raw = self.oanda.pricing.get(self.settings.get('account_id'
                                                             ), **args)
        price = json.loads(price_raw.raw_body)
        spread = abs(float(price.get('prices')[0].get('bids')[0].get('price'
                                                                     )) - float(price.get('prices')[0].get('asks'
                                                                                                           )[0].get(
            'price')))
        return spread

    def get_price(self, ins):
        args = {'instruments': ins}
        price_raw = self.oanda.pricing.get(self.settings.get('account_id'
                                                             ), **args)
        price_json = json.loads(price_raw.raw_body)
        price = (float(price_json.get('prices')[0].get('bids')[0].get('price'
                                                                      )) + float(price_json.get('prices')[0].get('asks'
                                                                                                                 )[
            0].get(
            'price'))) / 2.0
        return price

    def get_calendar_data(self, date):
        # extract event data regarding the current trading week
        df = {}
        currencies = ['CNY', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD', 'AUD', 'ALL']
        impacts = ['Non-Economic', 'Low Impact Expected', 'Medium Impact Expected', 'High Impact Expected']
        for curr in currencies:
            for impact in impacts:
                column_name = curr + impact
                column_name = column_name.replace(' ', '')
                df[column_name] = self.calendar.count(date=date, currency=curr, impact=impact)
        dt = datetime.datetime.strptime(date, '%Y-%m-%d')
        if dt.weekday() == 4:
            dt += datetime.timedelta(days=3)
        else:
            dt += datetime.timedelta(days=1)
        date_next = dt.strftime('%Y-%m-%d')
        for curr in currencies:
            for impact in impacts:
                column_name = curr + impact + '_next'
                column_name = column_name.replace(' ', '')
                df[column_name] = self.calendar.count(date=date_next, currency=curr, impact=impact)
        if dt.weekday() == 4:
            dt += datetime.timedelta(days=3)
        else:
            dt += datetime.timedelta(days=1)
        date_next = dt.strftime('%Y-%m-%d')
        for curr in currencies:
            for impact in impacts:
                column_name = curr + impact + '_next2'
                column_name = column_name.replace(' ', '')
                df[column_name] = self.calendar.count(date=date_next, currency=curr, impact=impact)
        return df

    def candles_to_db(self, candles, ins, completed=True, upsert=False):
        new_count = 0
        update_count = 0
        for candle in candles:
            if (not bool(candle.get('complete'))) and completed:
                continue
            time = candle.get('time')[:10]  # take the YYYY-MM-DD part
            candle_old = self.table.find_one(date=time, ins=ins)
            candle_new = {'ins': ins, 'date': time, 'open': candle.get('mid').get('o'),
                          'close': candle.get('mid').get('c'),
                          'high': candle.get('mid').get('h'), 'low': candle.get('mid').get('l'),
                          'volume': candle.get('volume'),
                          'complete': bool(candle.get('complete'))}
            if candle_old:
                if self.verbose > 1:
                    print(ins + ' ' + time + ' already in dataset')
                new_count += 1
                if upsert:
                    update_count += 1
                    self.table.upsert(candle_new, ['ins', 'date'])
                continue
            if self.verbose > 1:
                print('Inserting ' + str(candle_new))
            if self.verbose > 0:
                print('New Candles: ' + str(new_count) + ' | Updated Candles: ' + str(update_count))
            self.table.insert(candle_new)

    def get_candles(
            self,
            ins,
            granularity,
            num_candles):
        request = Request('GET',
                          '/v3/instruments/{instrument}/candles?count={count}&price={price}&granularity={granularity}'
                          )
        request.set_path_param('instrument', ins)
        request.set_path_param('count', num_candles)
        request.set_path_param('price', 'M')
        request.set_path_param('granularity', granularity)
        response = self.oanda.request(request)
        candles = json.loads(response.raw_body)
        return candles.get('candles')

    def get_market_df(self, date, inst, complete):
        drow = {'date': date}
        for ins in inst:
            if complete:
                icandle = self.table.find_one(date=date, ins=ins, complete=1)
            else:
                icandle = self.table.find_one(date=date, ins=ins)
            if not icandle:
                if self.verbose > 1:
                    print('Candle does not exist ' + ins + ' ' + str(date))
                drow[ins + '_vol'] = -999999
                drow[ins + '_open'] = -999999
                drow[ins + '_close'] = -999999
                drow[ins + '_high'] = -999999
                drow[ins + '_low'] = -999999
            else:
                drow[ins + '_vol'] = int(icandle['volume'])
                drow[ins + '_open'] = float(icandle['open'])
                if float(icandle['close']) > float(icandle['open']):
                    drow[ins + '_close'] = int(1)
                else:
                    drow[ins + '_close'] = int(-1)
                drow[ins + '_high'] = float(icandle['high']) - float(icandle['open'])
                drow[ins + '_low'] = float(icandle['low']) - float(icandle['open'])
        return drow

    def data2sheet(self, write_raw=False, write_predict=True, improve_model=False, maxdate=None, new_estim=False,
                   complete=True):
        inst = []
        if complete:
            c_cond = ' complete = 1'
        else:
            c_cond = ' complete in (0,1)'
        statement = 'select distinct ins from dailycandles order by ins;'
        for row in self.db.query(statement):
            inst.append(row['ins'])
        dates = []
        if maxdate:
            statement = 'select distinct date from dailycandles where date <= ' + maxdate + ' and ' + c_cond + ' order by date;'
        else:
            statement = 'select distinct date from dailycandles where ' + c_cond + ' order by date;'
        for row in self.db.query(statement):
            # if row['date'][:4] == year:
            dates.append(row['date'])
        df_dict = []
        if (not improve_model) and (not new_estim):  # if we want to read only it is enough to take the last days
            dates = dates[-4:]
        #dates = dates[-30:] # use this line to decrease computation time for development
        if self.verbose > 0:
            print('INFO: Starting data frame preparation')
            bar = progressbar.ProgressBar(maxval=len(dates),     widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
        index = 0
        for date in dates:
            if self.verbose > 0:
                bar.update(index)
            index += 1
            # check whether the candle is from a weekday
            date_split = date.split('-')
            weekday = int(datetime.datetime(int(date_split[0]), int(date_split[1]), int(date_split[2])).weekday())
            if weekday == 4 or weekday == 5:  # saturday starts on friday and sunday on saturday
                continue
            # start with the calendar data
            df_row = self.get_calendar_data(date)
            df_row['weekday'] = weekday
            today_df = self.get_market_df(date, inst, complete)
            # yest_df = self.get_market_df(prev_working_day(date), inst, complete)
            # yest_df.pop('date')  # remove the date key from prev day
            df_row = merge_dicts(df_row, today_df, '')
            # df_row = merge_dicts(df_row, yest_df, '_yester')
            df_dict.append(df_row)
        df = pd.DataFrame(df_dict)
        if self.verbose > 0:
            bar.finish()
        # code.interact(banner='', local=locals())
        if write_raw:
            print('Constructed DF with shape ' + str(df.shape))
            outname = '/home/ubuntu/data/cexport.csv'
            df.to_csv(outname)
        datecol = df['date'].copy()  # copy for usage in improveEstim
        df.drop(['date'], 1, inplace=True)
        prediction = {}
        if self.verbose > 0:
            print('INFO: Starting prediction')
            bar = progressbar.ProgressBar(maxval=len(df.columns),
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
        index = 0
        for col in df.columns:
            if self.verbose > 0:
                bar.update(index)
            index += 1
            parts = col.split('_')
            if len(parts) < 3:
                if self.verbose > 1:
                    print('WARNING: Unexpected column ' + col)
                continue
            if not ('_high' in col or '_low' in col or '_close' in col):
                continue
            if '_yester' in col:  # skip yesterday stuff for prediction
                continue
            if improve_model:
                self.improve_estimator(col, df, datecol)
            prediction_value, previous_value = self.predict_column(col, df, new_estimator=new_estim)
            instrument = parts[0] + '_' + parts[1]
            typ = parts[2]
            if instrument in prediction.keys():
                prediction[instrument][typ] = prediction_value  # store diff to prev day
            else:
                prediction[instrument] = {typ: prediction_value}
            if self.verbose > 1:
                print(col + ' ' + str(prediction_value))
        if self.verbose > 0:
            bar.finish()
        if write_predict:
            if complete:
                outfile = open(self.settings['prices_path'], 'w')
            else:
                outfile = open('{0}.partial'.format(self.settings['prices_path']),
                               'w')  # seperate file for partial estimates
            outfile.write('INSTRUMENT,HIGH,LOW,CLOSE\n')
            for instr in prediction.keys():
                outfile.write(str(instr) + ',' + str(prediction[instr].get('high')) + ',' + str(
                    prediction[instr].get('low')) + ',' + str(
                    prediction[instr].get('close')) + '\n')
            outfile.close()

    def get_feature_importances(self):
        feature_names = []
        statement = 'select distinct ins from dailycandles order by ins;'
        for row in self.db.query(statement):
            feature_names.append(row['ins'] + '_volume')
            feature_names.append(row['ins'] + '_open')
            feature_names.append(row['ins'] + '_close')
            feature_names.append(row['ins'] + '_high')
            feature_names.append(row['ins'] + '_low')
        sql = 'select distinct name from estimators;'
        for row in self.db.query(sql):
            pcol = row.get('name')
            try:
                estimator_path = self.settings.get('estim_path') + pcol
                estimator = EstimatorPipeline(path=estimator_path)
            except FileNotFoundError:
                print('Failed to load model for ' + pcol)
                continue
            if self.verbose > 1:
                print(pcol)
            for name, importance in zip(feature_names, estimator.get_feature_importances()):
                feature_importance = {'name': pcol, 'feature': name, 'importance': importance}
                self.importances.upsert(feature_importance, ['name', 'feature'])

    @staticmethod
    def dist_to_now(input_date):
        now = datetime.datetime.now()
        ida = datetime.datetime.strptime(input_date, '%Y-%m-%d')
        delta = now - ida
        return math.exp(-delta.days / 365.25)  # exponentially decaying weight decay

    def improve_estimator(self, pcol, df, datecol):
        try:
            estimator_name = self.settings.get('estim_path') + pcol
            estimator = EstimatorPipeline(path=estimator_name)
        except FileNotFoundError:
            print('Failed to load model for ' + pcol)
            return
        params = estimator.get_params()
        attribute_switch = math.floor(np.random.random() * 3)
        n_estimators_base = int(params.get('n_estimators'))
        if attribute_switch == 0:
            n_range = get_range_int(n_estimators_base, 0.1, lower=10)
        else:
            n_range = [n_estimators_base]
        n_minsample = params.get('min_samples_split')
        if attribute_switch == 1:
            minsample = get_range_int(n_minsample, 0.1, lower=2)
        else:
            minsample = [n_minsample]
        learning_rate = params.get('learning_rate')
        if attribute_switch == 2:
            n_learn = get_range_flo(learning_rate, 0.01, lower=0.0001, upper=1)
        else:
            n_learn = [learning_rate]
        percentile = params.get('percentile')
        # percentile is always considered because this might be the most crucial parameter
        n_perc = get_range_int(percentile, 0.1, 1, 100)
        parameters = {'n_estimators': n_range,
                      'min_samples_split': minsample,
                      'learning_rate': n_learn,
                      'percentile': n_perc}
        weights = np.array(datecol.apply(self.dist_to_now).values[:])
        x = np.array(df.values[:])
        y = np.array(df[pcol].values[:])  # make a deep copy to prevent data loss in future iterations
        weights = weights[1:]
        y = y[1:]  # drop first line
        x = x[:-1, :]  # drop the last line
        i = 0
        while i < y.shape[0]:
            if y[i] < -999990:  # missing values are marked with -999999
                weights = np.delete(weights, i)
                y = np.delete(y, i)
                x = np.delete(x, i, axis=0)
            else:
                i += 1
        if '_close' in pcol:
            base_estimator = EstimatorPipeline(classifier=True)
            y = np.array(y, dtype=int).round()
        else:
            base_estimator = EstimatorPipeline()
        score_str = 'neg_mean_absolute_error'
        gridcv = GridSearchCV(base_estimator, parameters, cv=3, iid=False, error_score='raise',
                              scoring=score_str)
        try:
            gridcv.fit(x, y, sample_weight=weights)
        except Exception as e:  # TODO narrow exception. It can fail due to too few dimensions
            print('FATAL: failed to compute ' + pcol + ' ' + str(e))
            return
        if self.verbose > 1:
            print('Improving Estimator for ' + pcol + ' ' + str(gridcv.best_params_) + ' score: ' + str(
                gridcv.best_score_))
        gridcv.best_estimator_.write_to_disk(estimator_name)
        estimator_score = {'name': pcol, 'score': gridcv.best_score_}
        self.estimtable.upsert(estimator_score, ['name'])

    def predict_column(self, predict_column, df, new_estimator=False):
        x = np.array(df.values[:])
        y = np.array(df[predict_column].values[:])  # make a deep copy to prevent data loss in future iterations
        vprev = y[-1]
        y = y[1:]  # drop first line
        xlast = x[-1, :]
        x = x[:-1, :]  # drop the last line
        if new_estimator:
            if '_close' in predict_column:
                estimator = EstimatorPipeline(classifier=True)  # GradientBoostingRegressor()
            else:
                estimator = EstimatorPipeline()  # GradientBoostingRegressor()
            i = 0
            while i < y.shape[0]:
                if y[i] < -999990:  # missing values are marked with -999999
                    y = np.delete(y, i)
                    x = np.delete(x, i, axis=0)
                else:
                    i += 1
            estimator.fit(x, y)
            estimator_name = self.settings.get('estim_path') + predict_column
            estimator.write_to_disk(estimator_name)
        else:
            estimator_name = self.settings.get('estim_path') + predict_column
            estimator = EstimatorPipeline(path=estimator_name)
        yp = estimator.predict(xlast.reshape(1, -1))
        return yp[0], vprev

    def get_units(self, dist, ins):
        # get the number of units to trade for a given pair
        trailing_currency = ''
        if dist == 0:
            return 0
        leading_currency = ins.split('_')[0]
        price = self.get_price(ins)
        # each trade should risk 1% of NAV at SL at most. Usually it will range
        # around 0.1 % - 1 % depending on expectation value
        target_exposure = self.settings.get('account_risk') * 0.01
        conversion = self.get_conversion(leading_currency)
        if not conversion:
            trailing_currency = ins.split('_')[1]
            conversion = self.get_conversion(trailing_currency)
            if conversion:
                conversion = conversion / price
        multiplier = min(price / dist, 100)  # no single trade can be larger than the account NAV
        if not conversion:
            print('CRITICAL: Could not convert ' + leading_currency + '_' + trailing_currency + ' to EUR')
            return 0  # do not place a trade if conversion fails
        raw_units = multiplier * target_exposure * conversion
        if raw_units > 0:
            return math.floor(raw_units)
        else:
            return math.ceil(raw_units)

    def get_conversion(self, leading_currency):
        # get conversion rate to account currency
        account_currency = 'EUR'
        # trivial case
        if leading_currency == account_currency:
            return 1
        # try direct conversion
        for ins in self.allowed_ins:
            if leading_currency in ins.name and account_currency in ins.name:
                price = self.get_price(ins.name)
                if ins.name.split('_')[0] == account_currency:
                    return price
                else:
                    return 1.0 / price
        # try conversion via usd
        eurusd = self.get_price('EUR_USD')
        if not eurusd:
            return None
        for ins in self.allowed_ins:
            if leading_currency in ins.name and 'USD' in ins.name:
                price = self.get_price(ins.name)
                if not price:
                    return None
                if ins.name.split('_')[0] == 'USD':
                    return price / eurusd
                else:
                    return 1.0 / (price * eurusd)
        return None

    def get_score(self, column_name):
        row = self.estimtable.find_one(name=column_name)
        if row:
            return row.get('score')
        else:
            if self.verbose > 0:
                print('WARNING: Unscored estimator - ' + column_name)
            return None

    def open_limit(self, ins, close_only=False, complete=True):
        if complete:
            df = pd.read_csv(self.settings['prices_path'])
        else:
            df = pd.read_csv('{0}.partial'.format(self.settings['prices_path']))
        op = self.get_price(ins)
        cl = df[df['INSTRUMENT'] == ins]['CLOSE'].values[0]
        hi = df[df['INSTRUMENT'] == ins]['HIGH'].values[0] + op
        lo = df[df['INSTRUMENT'] == ins]['LOW'].values[0] + op
        price = self.get_price(ins)
        # get the R2 of the consisting estimators
        column_name = ins + '_close'
        close_score = self.get_score(column_name)
        if not close_score:
            return
        column_name = ins + '_high'
        high_score = self.get_score(column_name)
        if not high_score:
            return
        column_name = ins + '_low'
        low_score = self.get_score(column_name)
        if not low_score:
            return

        spread = self.get_spread(ins)
        trade = None
        current_units = 0
        for tr in self.trades:
            if tr.instrument == ins:
                trade = tr
        if trade:
            is_open = True
            if close_score < -1:  # if we do not trust the estimator we should not move forward
                self.oanda.trade.close(self.settings.get('account_id'), trade.id)
            if trade.currentUnits * cl < 0:
                self.oanda.trade.close(self.settings.get('account_id'), trade.id)
                is_open = False
            if is_open:
                return
        if close_only:
            return  # if this flag is set only check for closing and then return
        if close_score < -1:
            return
        if cl > 0:
            step = 1.8 * abs(low_score)
            sl = lo - step
            entry = lo + spread / 2
            sldist = entry - sl
            tp1 = hi - abs(high_score) - spread / 2
            tp2 = hi - spread / 2
            tp3 = hi - abs(step) - spread / 2
        else:
            step = 1.8 * abs(high_score)
            sl = hi + step
            entry = hi - spread / 2
            sldist = sl - entry
            tp1 = lo + abs(low_score) + spread / 2
            tp2 = lo + spread / 2
            tp3 = lo + abs(step) + spread / 2
        rr = abs((tp2 - entry) / (sl - entry))
        if rr < 1.5:  # Risk-reward too low
            if self.verbose > 1:
                print(ins + ' RR: ' + str(rr) + ' | ' + str(entry) + '/' + str(sl) + '/' + str(tp2))
            return None
        # if you made it here its fine, lets open a limit order
        # r2sum is used to scale down the units risked to accomodate the estimator quality
        units = self.get_units(abs(sl - entry), ins) * min(abs(cl),
                                                           1.0)
        if units > 0:
            units = math.floor(units)
        if units < 0:
            units = math.ceil(units)
        if abs(units) < 1:
            return None  # oops, risk threshold too small
        if tp2 < sl:
            units *= -1
        pip_location = self.get_pip_size(ins)
        pip_size = 10**(-pip_location+1)
        if abs(sl - entry) < 200 * 10 ** (-pip_location):  # sl too small
            return None
        if (entry - price) * units > 0:
            otype = 'STOP'
        else:
            otype = 'LIMIT'
        format_string = '30.' + str(pip_location) + 'f'
        tp1 = format(tp1, format_string).strip()
        tp2 = format(tp2, format_string).strip()
        tp3 = format(tp3, format_string).strip()
        sl = format(sl, format_string).strip()
        sldist = format(sldist, format_string).strip()
        entry = format(entry, format_string).strip()
        expiry = datetime.datetime.now() + datetime.timedelta(days=1)
        units = int(units/3) # open three trades to spread out the risk
        for tp in [tp1, tp2, tp3]:
            args = {'order': {
                'instrument': ins,
                'units': units,
                'price': entry,
                'type': otype,
                'timeInForce': 'GTD',
                'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'takeProfitOnFill': {'price': tp, 'timeInForce': 'GTC'},
                'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'},
                'trailingStopLossOnFill': { 'distance': sldist, 'timeInForce': 'GTC'}
            }}
            #code.interact(banner='', local=locals())
            if self.verbose > 1:
                print(args)
            ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
            if self.verbose > 1:
                print(ticket.raw_body)
