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
import code

try:
    # noinspection PyUnresolvedReferences
    import v20
    # noinspection PyUnresolvedReferences
    from v20.request import Request

    v20present = True
except ImportError:
    print('WARNING: V20 library not present. Connection to broker not possible')
    v20present = False

# import code

import progressbar
import configparser
import math
import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV
import dataset
import pickle
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import re


def prev_working_day(_day):
    # Get the previous working day for this day. Since we work with NY candle alignment the week is sunday to thursday
    # _day: String representing the day in Format 2018-11-23

    date1 = datetime.datetime.strptime(_day, '%Y-%m-%d')
    wd = date1.weekday()
    if wd < 6:
        date2 = date1 - datetime.timedelta(days=1)
    else:
        date2 = date1 - datetime.timedelta(days=3)  # skip to friday on weekends
    return date2.strftime('%Y-%m-%d')


def merge_dicts(dict1, dict2, suffix):
    # Merge two dicts
    # dict1: this dict will keep all the key names as are
    # dict2: The keys of this dict will receive the suffix
    # suffix: suffix to append to the keys of dict2

    for key in dict2.keys():
        key_name = key + suffix
        if key_name in dict1.keys():
            raise ValueError('duplicate key {0} while merging'.format(key_name))
        dict1[key_name] = dict2[key]
    return dict1


def get_range_int(_val, change, lower=-math.inf, upper=math.inf):
    # Returns a range including a decremented and incremented integer value within the range
    # _val: Center value
    # change: relative distance of range values from center
    # lower: lower bound
    # upper: upper bound

    if _val > upper:
        val = upper
    else:
        val = _val
    lower_value = math.floor(val * (1 - change))
    upper_value = math.ceil(val * (1 + change))
    rang = []
    if lower <= lower_value < val:
        rang.append(lower_value)
    rang.append(val)
    if val < upper_value <= upper:
        rang.append(upper_value)
    return rang


def get_range_flo(val, change, lower=-math.inf, upper=math.inf):
    # Returns a range including a decremented and incremented floating value within the range
    # _val: Center value
    # change: relative distance of range values from center
    # lower: lower bound
    # upper: upper bound

    lower_value = val * (1 - change)
    upper_value = val * (1 + change)
    rang = []
    if lower <= lower_value < val:
        rang.append(lower_value)
    rang.append(val)
    if val < upper_value <= upper:
        rang.append(upper_value)
    return rang


def get_gb_importances(gb, x, y):
    # returns the feature_importances using the provided estimator
    # gb: Estimator
    # x: Features
    # y: target

    gb.fit(x, y)
    return gb.feature_importances_


class EstimatorPipeline(object):
    # Custom estimator class combining percentile selection with gradient boosting

    feature_importances_: object

    def __init__(self, percentile=5, n_components=0, learning_rate=0.1, n_estimators=100, min_samples_split=2, path=None,
                 classifier=False, input_dimension=150):
        # Class init
        # percentile: importance percentile of features to use
        # learning_rate: learning rate for the gradient boosting
        # n_estimators: Number of estimators for the gradient boosting
        # min_samples_split: Min. samples to split in the gradient boosting
        # path: if provided will attempt to load stored estimators
        # classifier: Whether to train a classifier and not a regressor

        self.classifier = classifier
        if path:  # read from disk
            gb_path = path + '.gb'
            self.gb = pickle.load(open(gb_path, 'rb'))
            percentile_path = path + '.pipe'
            percentile_attr = pickle.load(open(percentile_path, 'rb'))
            param_path = path + '.param'
            self.params = pickle.load(open(param_path, 'rb'))
            # score_func = lambda x, y: get_gb_importances(self.gb, x, y)
            score_func: Callable[[Any, Any], Any] = lambda x, y: get_gb_importances(self.gb, x, y)
            self.percentile = SelectPercentile(score_func=score_func, percentile=self.params.get('percentile'))
            self.percentile.scores_ = percentile_attr.get('scores')
            self.percentile.pvalues_ = percentile_attr.get('pvalues')
            pca_path = path+'.pca'
            try:
                self.pca = pickle.load(open(pca_path, 'rb'))
            except:
                self.pca = None
        else:
            self.params = {'percentile': percentile, 'learning_rate': learning_rate, 'n_estimators': n_estimators,
                           'min_samples_split': min_samples_split, 'n_components': n_components}
            if classifier:
                self.gb = GradientBoostingClassifier(learning_rate=learning_rate, min_samples_split=min_samples_split,
                                                     n_estimators=n_estimators)
            else:
                self.gb = GradientBoostingRegressor(learning_rate=learning_rate, min_samples_split=min_samples_split,
                                                    n_estimators=n_estimators)
            # score_func = lambda x, y: get_gb_importances(self.gb, x, y)
            score_func: Callable[[Any, Any], Any] = lambda x, y: get_gb_importances(self.gb, x, y)
            self.percentile = SelectPercentile(score_func=score_func, percentile=percentile)
            if n_components > 0:
                self.pca = PCA(n_components=int(input_dimension*n_components*percentile/100))
            else:
                self.pca = None
        if not self.pca:
            self.pipeline = make_pipeline(
                self.percentile,
                self.gb
            )
        else:
            self.pipeline = make_pipeline(
                self.percentile,
                self.pca,
                self.gb
            )

    def get_feature_importances(self):
        # provide feature importances of encapsuled gradient boosting

        return self.gb.feature_importances_

    def get_params(self, deep=True):  # keyword deep needed for gridsearch
        # this function is needed for the custom estimator that it can be passed to gridseearch

        return self.params

    def fit(self, x, y, sample_weight=None):
        # fit encapsuled pipeline
        # x: features
        # y: target
        # sample_weight: Optional sample weight

        x = np.nan_to_num(x)
        if self.classifier:
            self.pipeline.fit(x, y, gradientboostingclassifier__sample_weight=sample_weight)
        else:
            self.pipeline.fit(x, y, gradientboostingregressor__sample_weight=sample_weight)
        self.feature_importances_ = self.gb.feature_importances_
        return self

    def write_to_disk(self, path):
        # Write the different parts of the estimator to disk for later usage
        # path: Filename without any extensions

        gb_path = path + '.gb'
        pickle.dump(self.gb, open(gb_path, 'wb'))
        pca_path = path + '.pca'
        pickle.dump(self.pca, open(pca_path, 'wb'))
        # the select_percentile can not be pickles due to the choice of the importance func.
        # So we pickle the attributes
        pipe_path = path + '.pipe'
        pipe_attr = {'scores': self.percentile.scores_, 'pvalues': self.percentile.pvalues_}
        pickle.dump(pipe_attr, open(pipe_path, 'wb'))
        param_path = path + '.param'
        pickle.dump(self.params, open(param_path, 'wb'))

    def predict(self, x):
        # predict using pre trained estimator
        # x: features

        x = np.nan_to_num(x)
        try:
            return self.pipeline.predict(x)
        except Exception as e:
            print('Failed to predict ' + str(e))
            return [0,]

    def score(self, x, y=None, sample_weight=None):
        # Calculate accuracy score
        # x: Features
        # y: Targets
        # sample_weight: Optional sample weight

        x = np.nan_to_num(x)
        return self.pipeline.score(x, y=y, sample_weight=sample_weight)

    def set_params(self, percentile=None, learning_rate=None, n_estimators=None, min_samples_split=None, n_components=None):
        # Set the estimator parameters
        # percentile: Feature importance percentile used for prediction
        # learning_rate: Learning rate for gradient boosting
        # n_estimators: Number of estimators for gradient boosting
        # min_samples_split: Min Samples split for gradient boosting
        if n_components:
            self.pca.set_params(n_components=n_components)
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
    # This class controls most of the program flow for:
    # - getting the data
    # - constructing the data frame for training and prediction
    # - actual training and prediction of required models
    # - acting in the market based on the prediction

    def __init__(self, config_name, _type, verbose=2):
        # class init
        # config_name: Path to config file
        # _type: which section of the config file to use for broker connection
        # verbose: verbositiy. 0: Display FATAL only, 1: Display progress bars also, >=2: Display a lot of misc info

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
            self.orders = self.oanda.order.list(self.settings.get('account_id')).get('orders', '200')
        self.db = dataset.connect(config.get('data', 'candle_path'))
        self.calendar_db = dataset.connect(config.get('data', 'calendar_path'))
        self.calendar = self.calendar_db['calendar']
        self.table = self.db['dailycandles']
        self.estimtable = self.db['estimators']
        self.importances = self.db['feature_importances']
        # the following arrays are used to collect aggregate information in estimator improvement
        self.accuracy_array = []
        self.n_components_array = []

    def retrieve_data(self, num_candles, completed=True, upsert=False):
        # collect data for all available instrument from broker and store in database
        # num_candles: Number of candles, max. 500
        # completed: Whether to use only completed candles, in other words whether to ignore today
        # upsert: Whether to update existing entries

        for ins in self.allowed_ins:
            candles = self.get_candles(ins.name, 'D', num_candles)
            self.candles_to_db(candles, ins.name, completed=completed, upsert=upsert)

    def get_pip_size(self, ins):
        # Returns pip size for a given instrument
        # ins: Instrument, e.g. EUR_USD

        pip_loc = [_ins.pipLocation for _ins in self.allowed_ins if _ins.name == ins]
        if not len(pip_loc) == 1:
            return None
        return -pip_loc[0] + 1

    def get_spread(self, ins):
        # Returns spread for a instrument
        # ins: Instrument, e.g. EUR_USD

        if not v20present:
            return 0.00001
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
        # Returns price for a instrument
        # ins: Instrument, e.g. EUR_USD

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

    def strip_number(self, _number):
        # try to get a numeric value from a string like e.g. '3.4M'
        # _number: partly numeric string

        try:
            num = float(re.sub('[^0-9]', '', _number))
            if np.isnan(num):
                return 0
            return num
        except ValueError as e:
            if self.verbose > 2:
                print(str(e))
            return None

    def get_calendar_data(self, date):
        # extract event data regarding the current trading week
        # date: Date in format '2018-06-23'

        # the date is taken from oanda NY open alignment. Hence if we use only complete candles this date
        # will be the day before yesterday
        df = {}
        currencies = ['CNY', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD', 'AUD', 'ALL']
        impacts = ['Non-Economic', 'Low Impact Expected', 'Medium Impact Expected', 'High Impact Expected']
        for curr in currencies:
            # calculate how actual and forecast numbers compare. If no forecast available just use the previous number
            for impact in impacts:
                sentiment = 0
                for row in self.calendar.find(date=date, currency=curr, impact=impact):
                    actual = self.strip_number(row.get('actual'))
                    if not actual:
                        continue
                    forecast = self.strip_number(row.get('forecast'))
                    if forecast:
                        sentiment += math.copysign(1,
                                                   actual - forecast)
                        # (actual - forecast)/(abs(actual)+abs(forecast)+0.01)
                        continue
                    previous = self.strip_number(row.get('previous'))
                    if previous:
                        sentiment += math.copysign(1,
                                                   actual - previous)
                        # (actual-previous)/(abs(actual)+abs(previous)+0.01)
                column_name = curr + '_sentiment_' + impact
                df[column_name] = sentiment
            for impact in impacts:
                column_name = curr + impact
                column_name = column_name.replace(' ', '')
                df[column_name] = self.calendar.count(date=date, currency=curr, impact=impact)
        dt = datetime.datetime.strptime(date, '%Y-%m-%d')

        # when today is friday (4) skip the weekend, else go one day forward. Then we have reached yesterday
        if dt.weekday() == 4:
            dt += datetime.timedelta(days=3)
        else:
            dt += datetime.timedelta(days=1)
        date_next = dt.strftime('%Y-%m-%d')
        for curr in currencies:
            # calculate how actual and forecasted numbers compare. If no forecast available just use the previous number
            for impact in impacts:
                sentiment = 0
                for row in self.calendar.find(date=date, currency=curr, impact=impact):
                    actual = self.strip_number(row.get('actual'))
                    if not actual:
                        continue
                    forecast = self.strip_number(row.get('forecast'))
                    if forecast:
                        sentiment += math.copysign(1,
                                                   actual - forecast)
                        # (actual - forecast) / (abs(actual) + abs(forecast) + 0.01)
                        continue
                    previous = self.strip_number(row.get('previous'))
                    if previous:
                        sentiment += math.copysign(1,
                                                   actual - previous)
                        # (actual - previous) / (abs(actual) + abs(previous) + 0.01)
                column_name = curr + '_sentiment_' + impact + '_next'
                df[column_name] = sentiment
            for impact in impacts:
                column_name = curr + impact + '_next'
                column_name = column_name.replace(' ', '')
                df[column_name] = self.calendar.count(date=date_next, currency=curr, impact=impact)
        # when today is friday (4) skip the weekend, else go one day forward. Then we have reached today
        if dt.weekday() == 4:
            dt += datetime.timedelta(days=3)
        else:
            dt += datetime.timedelta(days=1)
        date_next = dt.strftime('%Y-%m-%d')
        for curr in currencies:
            # calculate how actual and forecasted numbers compare.
            #  If no forecast available just use the previous number
            for impact in impacts:
                sentiment = 0
                for row in self.calendar.find(date=date, currency=curr, impact=impact):
                    forecast = self.strip_number(row.get('forecast'))
                    if not forecast:
                        continue
                    previous = self.strip_number(row.get('previous'))
                    if previous:
                        sentiment += math.copysign(1,
                                                   forecast - previous)
                        # (forecast-previous)/(abs(forecast)+abs(previous)+0.01)
                column_name = curr + '_sentiment_' + impact + '_next2'
                df[column_name] = sentiment
            for impact in impacts:
                column_name = curr + impact + '_next2'
                column_name = column_name.replace(' ', '')
                df[column_name] = self.calendar.count(date=date_next, currency=curr, impact=impact)
        return df

    def candles_to_db(self, candles, ins, completed=True, upsert=False):
        # Write candles to sqlite database
        # candles: Array of candles
        # ins: Instrument, e.g. EUR_USD
        # completed: Whether to write only completed candles
        # upsert: Whether to update if a dataset exists

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

    def get_candles(self, ins, granularity, num_candles):
        # Get pricing data in candle format from broker
        # ins: Instrument
        # granularity: Granularity as in 'H1', 'H4', 'D', etc
        # num_candles: Number of candles, max. 500

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

    def get_market_df(self, date, inst, complete, bootstrap=False):
        # Create Market data portion of data frame
        # date: Date in Format 'YYYY-MM-DD'
        # inst: Array of instruments
        # complete: Whether to use only complete candles
        if bootstrap:
            bs_flag = 1
        else:
            bs_flag = 0
        data_frame = {'date': date}
        for ins in inst:
            if complete:
                candle = self.table.find_one(date=date, ins=ins, complete=1)
            else:
                candle = self.table.find_one(date=date, ins=ins)
            if not candle:
                if self.verbose > 1:
                    print('Candle does not exist ' + ins + ' ' + str(date))
                data_frame[ins + '_vol'] = -999999
                data_frame[ins + '_open'] = -999999
                data_frame[ins + '_close'] = -999999
                data_frame[ins + '_high'] = -999999
                data_frame[ins + '_low'] = -999999
            else:
                spread = self.get_spread(ins)
                volume = float(candle['volume']) * (1 + np.random.normal() * 0.001 * bs_flag)  # 0.1% deviation
                open = float(candle['open'])+spread*np.random.normal()*0.5 * bs_flag
                close = float(candle['close']) + spread * np.random.normal() * 0.5 * bs_flag
                high = float(candle['high']) + spread * np.random.normal() * 0.5 * bs_flag
                low = float(candle['low']) + spread * np.random.normal() * 0.5 * bs_flag
                data_frame[ins + '_vol'] = int(volume)
                data_frame[ins + '_open'] = float(open)
                if float(close) > float(open):
                    data_frame[ins + '_close'] = int(1)
                else:
                    data_frame[ins + '_close'] = int(-1)
                data_frame[ins + '_high'] = float(high) - float(open)
                data_frame[ins + '_low'] = float(low) - float(open)
        return data_frame

    def get_df_for_date(self, date, inst, complete, bootstrap=False):
        # Creates a dict containing all fields for the given date
        # date: Date to use in format 'YYYY-MM-DD'
        # inst: Array of instruments to use as inputs
        # complete: Whether to use only complete candles

        date_split = date.split('-')
        weekday = int(datetime.datetime(int(date_split[0]), int(date_split[1]), int(date_split[2])).weekday())
        if weekday == 4 or weekday == 5:  # saturday starts on friday and sunday on saturday
            return None
        # start with the calendar data
        df_row = self.get_calendar_data(date)
        df_row['weekday'] = weekday
        today_df = self.get_market_df(date, inst, complete, bootstrap=bootstrap)
        # yest_df = self.get_market_df(prev_working_day(date), inst, complete)
        # yest_df.pop('date')  # remove the date key from prev day
        return merge_dicts(df_row, today_df, '')

    def data2sheet(self, write_raw=False, write_predict=True, improve_model=False, maxdate=None,
                   complete=True, read_raw=False, close_only=False):
        # This method will take the input collected from oanda and forexfactory and merge in a Data Frame
        # write_raw: Write data frame used for training to disk
        # read_raw: Read data frame used for training from disk
        # write_predict: Write prediction file to disk to use it for trading later on
        # improve_model: Perform Hyper parameter improvement for the estimators
        # maxdate: Maximum data to use in the prediction. If None use all.
        # new_estim: Build new estimators with new Hyper parameters

        raw_name = '../data/cexport.csv'
        if read_raw:
            df = pd.read_csv(raw_name)
        else:
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
                date = row['date']
                date_split = date.split('-')
                weekday = int(datetime.datetime(int(date_split[0]), int(date_split[1]), int(date_split[2])).weekday())
                if weekday == 4 or weekday == 5:  # saturday starts on friday and sunday on saturday
                    continue
                dates.append(row['date'])
            df_dict = []
            if (not improve_model):  # if we want to read only it is enough to take the last days
                dates = dates[-3:]
            #dates = dates[-100:] # use this line to decrease computation time for development
            bar = None
            if self.verbose > 0:
                print('INFO: Starting data frame preparation')
                bar = progressbar.ProgressBar(maxval=len(dates),
                                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()
            index = 0
            num_samples = 1
            if improve_model:
                num_samples = 10
            for date in dates:
                if self.verbose > 0:
                    bar.update(index)
                index += 1
                for i in range(num_samples):  # 2 fold over sampling
                    df_row = self.get_df_for_date(date, inst, complete, bootstrap=improve_model)
                    # df_row = merge_dicts(df_row, yest_df, '_yester')
                    if df_row:
                        df_dict.append(df_row)
            df = pd.DataFrame(df_dict)
            if self.verbose > 0:
                bar.finish()
        if write_raw:
            print('Constructed DF with shape ' + str(df.shape))
            df.to_csv(raw_name, index=False)
        date_column = df['date'].copy()  # copy for usage in improveEstim
        df.drop(['date'], 1, inplace=True)
        prediction = {}
        bar = progressbar.ProgressBar(maxval=len(df.columns),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        if self.verbose > 0:
            print('INFO: Starting prediction')
            bar.start()
        index = 0
        for col in df.columns:
            if self.verbose > 0:
                bar.update(index)
            index += 1
            parts = col.split('_')
            if len(parts) < 3:
                if self.verbose > 2:
                    print('WARNING: Unexpected column ' + col)
                continue
            if not ('_high' in col or '_low' in col or '_close' in col):
                continue
            if close_only and not ('_close' in col):
                continue
            if '_yester' in col:  # skip yesterday stuff for prediction
                continue
            if improve_model:
                self.improve_estimator(col, df, date_column)
            prediction_value, previous_value = self.predict_column(col, df)
            instrument = parts[0] + '_' + parts[1]
            typ = parts[2]
            if instrument in prediction.keys():
                prediction[instrument][typ] = prediction_value  # store diff to prev day
            else:
                prediction[instrument] = {typ: prediction_value}
            if self.verbose > 1:
                print(col + ' ' + str(prediction_value))
        if improve_model and self.verbose > 0:
            print('Final Model accuracy: Mean: ' + str(np.mean(self.accuracy_array)) + ' Min: ' + str(
                np.min(self.accuracy_array)) + ' Max: ' + str(np.max(self.accuracy_array)))
            print('Percentile: ' + str(np.min(self.n_components_array)) + '/' + str(np.mean(self.n_components_array)) + '/' + str(np.max(self.n_components_array)))
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
        # Writes feature importances for all trained estimators to sqlite for further inspection

        inst = []
        dates = []
        statement = 'select distinct ins from dailycandles order by ins;'
        for row in self.db.query(statement):
            inst.append(row['ins'])
        statement = 'select distinct date from dailycandles where complete = 1 order by date;'
        for row in self.db.query(statement):
            # if row['date'][:4] == year:
            dates.append(row['date'])
        dates = dates[-10:]
        df_all = []
        for date in dates:
            df_dict = self.get_df_for_date(date, inst, True)
            if not df_dict:
                continue
            df_all.append(df_dict)
        df = pd.DataFrame(df_all)
        feature_names = df.columns
        sql = 'select distinct name from estimators;'
        for row in self.db.query(sql):
            pcol = row.get('name')
            try:
                estimator_path = self.settings.get('estim_path') + pcol
                estimator = pickle.load(open(estimator_path ,'rb')) # EstimatorPipeline(path=estimator_path)
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
        estimator_name = self.settings.get('estim_path') + pcol
        #estimator = NormalizedMLP(path=estimator_name)
        weights = np.array(datecol.apply(self.dist_to_now).values[:])
        x = np.array(df.values[:])
        y = np.array(df[pcol].values[:])  # make a deep copy to prevent data loss in future iterations
        parameters = {
            'learning_rate': [0.01],
            'n_estimators': [500],
            'subsample': [1.0],
            'min_samples_split': [2],
            'max_depth': [3],
            'max_features': [None],
            'max_leaf_nodes': [None]
        }
        weights = weights[1:]
        y = y[1:]  # drop first line
        x = x[:-1, :]  # drop the last line
        i = 0
        while i < y.shape[0]:
            if y[i] < -999990 or np.isnan(y[i]):  # missing values are marked with -999999
                weights = np.delete(weights, i)
                y = np.delete(y, i)
                x = np.delete(x, i, axis=0)
            else:
                i += 1
        # training data enhancement
#        x_enhance = []
#        y_enhance = []
#        for i in range(y.shape[0]):
            # add variations in target by +/- 1%
#            y_new = y[i]*1.01
#            x_new = x[i, :]
#            y_enhance.append(y_new)
#            x_enhance.append(x_new)
#            y_new = y[i] * 0.99
#            x_new = x[i, :]
#            y_enhance.append(y_new)
#            x_enhance.append(x_new)
        #x = np.append(x, x_enhance, axis=1)
        #y = np.append(y, y_enhance, axis=1)
        is_cla = False
        if '_close' in pcol:
            is_cla = True
        base_estimator = GradientBoostingRegressor() #EstimatorPipeline(input_dimension=x.shape[1])
        score_str = 'neg_mean_absolute_error'
        gridsearch_cv = GridSearchCV(base_estimator, parameters, cv=3, iid=False, error_score=-999999,
                                     scoring=score_str, verbose=1)
        try:
            x = np.nan_to_num(x)
            gridsearch_cv.fit(x, y, sample_weight=weights)
        except Exception as e:  # TODO narrow exception. It can fail due to too few dimensions
            print('FATAL: failed to compute ' + pcol + ' ' + str(e))
            return
        if self.verbose > 1:
            if '_close' in pcol:
                self.accuracy_array.append(gridsearch_cv.best_score_)
                print('Mean: ' + str(np.mean(self.accuracy_array)) + ' Min: ' + str(
                    np.min(self.accuracy_array)) + ' Max: ' + str(np.max(self.accuracy_array)))
                self.n_components_array.append(gridsearch_cv.best_params_.get('n_components'))
            print('Improving Estimator for ' + pcol + ' ' + str(gridsearch_cv.best_params_) + ' score: ' + str(
                gridsearch_cv.best_score_))
        #gridsearch_cv.best_estimator_.write_to_disk(estimator_name)
        pickle.dump(gridsearch_cv.best_estimator_, open(estimator_name, 'wb'))
        estimator_score = {'name': pcol, 'score': gridsearch_cv.best_score_}
        self.estimtable.upsert(estimator_score, ['name'])

    def predict_column(self, predict_column, df):
        # Predict the next outcome for a given column
        # predict_column: Columns to predict
        # df: Data Frame containing the column itself as well as any features
        # new_estimator: Whether to lead the existing estimator from disk or create a new one

        x = np.array(df.values[:])
        y = np.array(df[predict_column].values[:])  # make a deep copy to prevent data loss in future iterations
        vprev = y[-1]
        xlast = x[-1, :]
        estimator_name = self.settings.get('estim_path') + predict_column
        estimator = pickle.load(open(estimator_name ,'rb'))# EstimatorPipeline(path=estimator_name)
        yp = estimator.predict(xlast.reshape(1, -1))
        return yp[0], vprev

    def get_units(self, dist, ins):
        # get the number of units to trade for a given pair
        # dist: Distance to the SL
        # ins: Instrument to trade, e.g. EUR_USD

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
        # leading_currency: ISO Code of the leading currency for the traded pair

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
        # retrieves training score for given estimator
        # column_name: Name of the columns the estimator is predicting

        row = self.estimtable.find_one(name=column_name)
        if row:
            return row.get('score')
        else:
            if self.verbose > 0:
                print('WARNING: Unscored estimator - ' + column_name)
            return None

    def open_limit(self, ins, close_only=False, complete=True):
        # Open orders and close trades using the predicted market movements
        # close_only: Set to true to close only without checking for opening Orders
        # complete: Whether to use only complete candles, which means to ignore the incomplete candle of today

        if complete:
            df = pd.read_csv(self.settings['prices_path'])
        else:
            df = pd.read_csv('{0}.partial'.format(self.settings['prices_path']))
        candles = self.get_candles(ins, 'D', 1)
        candle = candles[0]
        op = float(candle.get('mid').get('o'))
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
        trades = []
        current_units = 0
        for tr in self.trades:
            if tr.instrument == ins:
                trades.append(tr)
        if len(trades) > 0:
            is_open = True
            if close_score < -1:  # if we do not trust the estimator we should not move forward
                for trade in trades:
                    self.oanda.trade.close(self.settings.get('account_id'), trade.id)
            for trade in trades:
                if trade.currentUnits * cl < 0:
                    self.oanda.trade.close(self.settings.get('account_id'), trade.id)
                    is_open = False
            if is_open:
                return
        if close_only:
            return
        if close_score < -1:
            return
        if cl > 0:
            step = 1.5 * abs(low_score)
            sl = lo - step
            entry = max([price + 2 * spread, lo])
            sldist = entry - sl + spread
            tp2 = hi
            tpstep = (tp2 - price) / 3
            tp1 = hi - 2 * tpstep
            tp3 = hi - tpstep
        else:
            step = 1.5 * abs(high_score)
            sl = hi + step
            entry = min([price - 2 * spread, hi])
            sldist = sl - entry + spread
            tp2 = lo
            tpstep = (price - tp2) / 3
            tp1 = lo + 2 * tpstep
            tp3 = lo + tpstep
        rr = abs((tp2 - entry) / (sl - entry))
        if rr < 1.5:  # Risk-reward too low
            if self.verbose > 1:
                print(ins + ' RR: ' + str(rr) + ' | ' + str(entry) + '/' + str(sl) + '/' + str(tp2))
            return None
        # if you made it here its fine, lets open a limit order
        # r2sum is used to scale down the units risked to accomodate the estimator quality
        units = self.get_units(abs(sl - entry), ins) * min(abs(cl),
                                                           1.0) * (1 + close_score)
        if units > 0:
            units = math.floor(units)
        if units < 0:
            units = math.ceil(units)
        if abs(units) < 1:
            return None  # oops, risk threshold too small
        if tp2 < sl:
            units *= -1
        pip_location = self.get_pip_size(ins)
        pip_size = 10 ** (-pip_location + 1)
        if abs(sl - entry) < 200 * 10 ** (-pip_location):  # sl too small
            return None
        # otype = 'MARKET'
        otype = 'STOP'
        format_string = '30.' + str(pip_location) + 'f'
        tp1 = format(tp1, format_string).strip()
        tp2 = format(tp2, format_string).strip()
        tp3 = format(tp3, format_string).strip()
        sl = format(sl, format_string).strip()
        sldist = format(sldist, format_string).strip()
        entry = format(entry, format_string).strip()
        expiry = datetime.datetime.now() + datetime.timedelta(minutes=10)
        # units = int(units/3) # open three trades to spread out the risk
        if abs(units) < 1:
            return
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
                'trailingStopLossOnFill': {'distance': sldist, 'timeInForce': 'GTC'}
            }}
            if self.verbose > 1:
                print(args)
            ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
            if self.verbose > 1:
                print(ticket.raw_body)
