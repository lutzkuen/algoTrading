#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Lutz Künneke"
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Lutz Künneke"
__email__ = "lutz.kuenneke89@gmail.com"
__status__ = "Working Prototype"
"""
Candle logger and ML controller
Use at own risk
Author: Lutz Kuenneke, 26.07.2018
"""

import configparser
import datetime
import json
import math
import re
import code
import time

import dataset
import numpy as np
import pandas as pd
import progressbar

try:
    # from observer import estimator as estimator
    from observer import estimator_cython as estimator
except ImportError:
    from observer import estimator as estimator

try:
    # noinspection PyUnresolvedReferences
    import v20
    # noinspection PyUnresolvedReferences
    from v20.request import Request
    v20present = True
except ImportError:
    print('WARNING: V20 library not present. Connection to broker not possible')
    v20present = False


def merge_dicts(dict1, dict2, suffix):
    """
    :param dict1: this dict will keep all the key names as are
    :param dict2: The keys of this dict will receive the suffix
    :param suffix: suffix to append to the keys of dict2
    :return: dict containing all fields of dict1 and dict2
    """

    for key in dict2.keys():
        key_name = key + suffix
        if key_name in dict1.keys():
            raise ValueError('duplicate key {0} while merging'.format(key_name))
        dict1[key_name] = dict2[key]
    return dict1


class Controller(object):
    """
    This class controls most of the program flow for:
    - getting the data
    - constructing the data frame for training and prediction
    - actual training and prediction of required models
    - acting in the market based on the prediction
    """

    def __init__(self, config_name, _type, verbose=2, write_trades=False):
        """
        class init

        Parameters
        ------
        config_name: Path to config file
        _type: which section of the config file to use for broker connection

        Keyword Parameters
        verbose: verbositiy. 0: Display FATAL only, 1: Display progress bars also, >=2: Display a lot of misc info
        write_trades: Whether to write all performed trades to a file
        """

        config = configparser.ConfigParser()
        config.read(config_name)
        self.verbose = verbose

        self.settings = {'estim_path': config.get('data', 'estim_path'),
                         'prices_path': config.get('data', 'prices_path'),
                         'raw_name': config.get('data', 'raw_name'),
                         'keras_path': config.get('data', 'keras_path')}
        self.write_trades = write_trades
        if write_trades:
            trades_path = config.get('data', 'trades_path')
            self.trades_file = open(trades_path, 'w')

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
        self.tradeable_instruments = ['EUR_USD', 
                                      'USD_JPY', 
                                      'USD_CHF', 
                                      'GBP_USD', 
                                      'GBP_JPY', 
                                      'AUD_JPY', 
                                      'USD_CAD',
                                      'NZD_USD',
                                      'AUD_USD',
                                      'EUR_JPY',
                                      'EUR_CHF',
                                      'EUR_AUD',
                                      'EUR_CAD',
                                      'EUR_NZD',
                                      'CHF_JPY',
                                      'CAD_JPY',
                                      'NZD_JPY',
                                      'GBP_CHF',
                                      'AUD_CHF',
                                      'CAD_CHF',
                                      'NZD_CHF',
                                      'GBP_AUD',
                                      'GBP_CAD',
                                      'GBP_NZD',
                                      'AUD_CAD',
                                      'AUD_NZD',
                                      'NZD_CAD',
                                      'EUR_GBP']
        self.db = dataset.connect(config.get('data', 'candle_path'))
        self.calendar_db = dataset.connect(config.get('data', 'calendar_path'))
        self.calendar = self.calendar_db['calendar']
        self.table = self.db['dailycandles']
        self.estimtable = self.db['estimators']
        self.importances = self.db['feature_importances']
        self.spread_db = dataset.connect(config.get('data', 'spreads_path'))
        self.spread_table = self.spread_db['spreads']
        self.prediction_db = dataset.connect(config.get('data', 'predictions_path'))
        self.prediction_table = self.prediction_db['prediction']
        # the following arrays are used to collect aggregate information in estimator improvement
        self.spreads = {}
        self.prices = {}

    def retrieve_data(self, num_candles, completed=True, upsert=False):
        """
        collect data for all available instrument from broker and store in database
        Parametes
        ------
        num_candles: Number of candles, max. 500

        Keyword arguments
        ------
        completed: Whether to use only completed candles, in other words whether to ignore today
        upsert: Whether to update existing entries

        Returns
        ------
        None
        """

        for ins in self.allowed_ins:
            candles = self.get_candles(ins.name, 'D', num_candles)
            self.candles_to_db(candles, ins.name, completed=completed, upsert=upsert)

    def get_pip_size(self, ins):
        """
        Returns pip size for a given instrument
        Parameters
        ------
        ins: Instrument, e.g. EUR_USD

        Returns
        ------
        int Location of the Pipsize after decimal point
        """

        pip_loc = [_ins.pipLocation for _ins in self.allowed_ins if _ins.name == ins]
        if not len(pip_loc) == 1:
            return None
        return -pip_loc[0] + 1

    def get_bidask(self, ins):
        # Returns spread for a instrument
        # ins: Instrument, e.g. EUR_USD

        args = {'instruments': ins}
        success = False
        while not success:
            try:
                price_raw = self.oanda.pricing.get(self.settings.get('account_id'), **args)
                success = True
            except Exception as e:
                print(str(e))
                time.sleep(1)
        price = json.loads(price_raw.raw_body)
        return (float(price.get('prices')[0].get('bids')[0].get('price'
                                                                )), float(price.get('prices')[0].get('asks'
                                                                                                     )[0].get(
            'price')))

    def save_spreads(self):
        """
        This function will save the spreads as seen in the market right now
        """
        for ins in self.allowed_ins:
            spread = self.get_spread(ins.name, spread_type='current')
            now = datetime.datetime.now()
            spread_object = {'timestamp': now, 'instrument': ins.name, 'weekday': now.weekday(), 'hour': now.hour,
                             'spread': spread}
            print(spread_object)
            self.spread_table.insert(spread_object)

    def get_spread(self, ins, spread_type='current'):
        """
        this function is a dispatcher the spread calculators
        current: Get the spread for the instrument at market
        worst: Get the worst ever recorded spread.
        weekend: Get the weekend spread
        mean: Get the mean spread for the given instrument the indicated hour and day
        """
        if spread_type == 'current':
            return self.get_current_spread(ins)
        elif spread_type == 'worst':
            return self.get_worst_spread(ins)
        elif spread_type == 'trading':
            return self.get_trading_spread(ins)

    def get_worst_spread(self, ins):
        """
        Return the worst ever recorded spread for the given instrument
        """
        max_spread = self.spread_db.query(
            "select max(spread) as ms from spreads where instrument = '{ins}';".format(ins=ins))
        for ms in max_spread:
            return float(ms['ms'])
        print('WARNING: Fall back to current spread')
        return self.get_current_spread(ins)

    def get_trading_spread(self, ins):
        """
        Returns the mean spread as observed during normal business hours
        """
        max_spread = self.spread_db.query(
            "select avg(spread) as ms from spreads where instrument = '{ins}' and weekday in (0, 1, 2, 3, 4) and hour > 6 and hour < 20;".format(ins=ins))
        for ms in max_spread:
            return float(ms['ms'])
        print('WARNING: Fall back to current spread')
        return self.get_current_spread(ins)

    def get_current_spread(self, ins):
        """
        Returns spread for a instrument
        ins: Instrument, e.g. EUR_USD
        """

        if not v20present:
            return 0.00001
        if ins in self.spreads.keys():
            return self.spreads[ins]
        args = {'instruments': ins}
        success = False
        while not success:
            try:
                price_raw = self.oanda.pricing.get(self.settings.get('account_id'), **args)
                success = True
            except Exception as e:
                print('Failed to get price ' + str(e))
                time.sleep(1)
        price = json.loads(price_raw.raw_body)
        spread = abs(float(price.get('prices')[0].get('bids')[0].get('price'
                                                                     )) - float(price.get('prices')[0].get('asks'
                                                                                                           )[0].get(
            'price')))
        self.spreads[ins] = spread
        return spread

    def get_price(self, ins):
        """
        Returns price for a instrument
        ins: Instrument, e.g. EUR_USD
        """

        try:
            args = {'instruments': ins}
            if ins in self.prices.keys():
                return self.prices[ins]
            price_raw = self.oanda.pricing.get(self.settings.get('account_id'
                                                                 ), **args)
            price_json = json.loads(price_raw.raw_body)
            price = (float(price_json.get('prices')[0].get('bids')[0].get('price'
                                                                          )) + float(price_json.get('prices')[0].get('asks'
                                                                                                                     )[
                0].get(
                'price'))) / 2.0
            self.prices[ins] = price
            return price
        except Exception as e:
            print(ins + 'get price ' + str(e))
            return None

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
        currencies = ['CNY', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'USD', 'AUD']  # , 'ALL']
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
                    print(candle_new)
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

    def get_moving_average(self, date, inst, period):
        """
        :param date: Date for which to calculate the MA. This date is not included in the calculation itself
        :param inst: Instrument Code
        :param period: Period in Days
        :return: Gives the MA of Daily Mean price as float
        """
        # code.interact(banner='', local=locals())
        query = "select avg((open+close+high+low)/4) as ma from dailycandles where ins = '" + inst + "' and (julianday('" + date + "') - julianday(date)) <= " + str(period) + " and date < '" + date + "';"
        for line in self.db.query(query):
            # code.interact(banner='', local=locals())
            if line['ma']:
                return float(line['ma'])
            else:
                return -999999

    def get_rsi(self, date, inst, period):
        """
        :param date:
        :param inst:
        :param period:
        :return:
        """
        prev_price = None
        avg_gain = 0
        avg_loss = 0
        query = "select (open+close+high+low)/4 as mean_price from dailycandles where ins = '" + inst + "' and (julianday('" + date + "') - julianday(date)) <= " + str(period) + " and date < '" + date + "';"
        for line in self.db.query(query):
            # code.interact(banner='', local=locals())
            if line['mean_price']:
                new_price = line['mean_price']
                if prev_price:
                    if new_price > prev_price:
                        avg_gain += new_price - prev_price
                    else:
                        avg_loss += prev_price - new_price
                prev_price = new_price
        if avg_loss == 0:
            # print(inst)
            # code.interact(banner='', local=locals())
            return 100
        rsival = 100 - 100 / ( 1 + avg_gain / avg_loss)
        # print('Calculated RSI ' + str(rsival))
        return rsival

    def get_market_df(self, date, inst, complete, bootstrap=False):
        # Create Market data portion of data frame
        # date: Date in Format 'YYYY-MM-DD'
        # inst: Array of instruments
        # complete: Whether to use only complete candles
        if bootstrap:
            bs_flag = 1
        else:
            bs_flag = 0
        # print('bsflag ' + str(bs_flag))
        data_frame = {'date': date}
        for ins in inst:
            if complete:
                candle = self.table.find_one(date=date, ins=ins, complete=1)
            else:
                candle = self.table.find_one(date=date, ins=ins)
            if not candle:
                if self.verbose > 2:
                    print('Candle does not exist ' + ins + ' ' + str(date))
                data_frame[ins + '_vol'] = -999999
                data_frame[ins + '_open'] = -999999
                data_frame[ins + '_close'] = -999999
                data_frame[ins + '_high'] = -999999
                data_frame[ins + '_low'] = -999999
                data_frame[ins + '_ma30'] = -999999
                data_frame[ins + '_ma60'] = -999999
                data_frame[ins + '_rsi14'] = -999999
                data_frame[ins + '_madiff'] = -999999
            else:
                spread = self.get_spread(ins, spread_type='trading')
                volume = float(candle['volume'])
                _open = float(candle['open'])
                close = float(candle['close'])
                high = float(candle['high'])
                low = float(candle['low'])
                data_frame[ins + '_vol'] = int(volume)
                data_frame[ins + '_open'] = float(_open)
                ma30 = self.get_moving_average(date, ins, 30)
                ma60 = self.get_moving_average(date, ins, 60)
                rsi14 = self.get_rsi(date, ins, 14)
                if float(close) > float(_open):
                    div = float(high) - float(_open)
                    if div > 0.000001:
                        data_frame[ins + '_close'] = (float(close) - float(_open))/div
                    else:
                        data_frame[ins + '_close'] = 0
                else:
                    div = float(_open) - float(low)
                    if div > 0.000001:
                        data_frame[ins + '_close'] = (float(close) - float(_open))/div
                    else:
                        data_frame[ins + '_close'] = 0
                data_frame[ins + '_high'] = float(high) - float(_open)
                data_frame[ins + '_low'] = float(low) - float(_open)
                data_frame[ins+'_ma30'] = _open - ma30
                data_frame[ins + '_ma60'] = _open - ma60
                data_frame[ins + '_madiff'] = ma60 - ma30
                data_frame[ins + '_rsi14'] = rsi14
                # print(rsi14)
        return data_frame

    def get_derived_features(self, today_df):
        derived_features = dict()
        # get the WTI to BCO ratio
        derived_features['BCO_over_WTI'] = today_df['WTICO_USD_open'] / today_df['BCO_USD_open']
        return derived_features


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
        additional_columns = self.get_derived_features(today_df)
        return merge_dicts(merge_dicts(df_row, today_df, ''), additional_columns, '')

    def predict_tomorrow(self):
        # first step is to get the actual tomorrow day
        now = datetime.datetime.now()
        oneday = datetime.timedelta(hours=24)
        twoday = datetime.timedelta(hours=48)
        threeday = datetime.timedelta(hours=72)
        weekday = now.weekday()
        if ( weekday < 5 ):
            prediction_day = now - oneday
        elif ( weekday == 5 ):
            prediction_day = now - twoday
        elif ( weekday == 6 ):
            prediction_day = now - threeday
        else:
            prediction_day = now - oneday
            print('What a weird day ' + str(weekday))
        prediction_day = prediction_day.strftime('%Y-%m-%d')
        print('Starting prediction based on ' + prediction_day)
        inst = []
        statement = 'select distinct ins from dailycandles order by ins;'
        for row in self.db.query(statement):
            inst.append(row['ins'])
        complete = False
        df_row = self.get_df_for_date(prediction_day, inst, complete, bootstrap=False)  # improve_model)
        df = pd.DataFrame([df_row])
        date_column = df['date'].copy()  # copy for usage in improveEstim
        df.drop(['date'], 1, inplace=True)
        prediction = {}
        for col in df.columns:
            parts = col.split('_')
            if len(parts) < 3:
                if self.verbose > 2:
                    print('WARNING: Unexpected column ' + col)
                continue
            if not ('_high' in col or '_low' in col or '_close' in col):
                continue
            if '_yester' in col:  # skip yesterday stuff for prediction
                continue
            prediction_value = self.predict_column(col, df)
            instrument = parts[0] + '_' + parts[1]
            typ = parts[2]
            if instrument in prediction.keys():
                prediction[instrument][typ] = prediction_value  # store diff to prev day
            else:
                prediction[instrument] = {typ: prediction_value}
            if self.verbose > 1:
                print(col + ' ' + str(prediction_value))
        outfile = open(self.settings['prices_path'], 'w')
        outfile.write('INSTRUMENT,HIGH,LOW,CLOSE\n')
        for instr in prediction.keys():
            outfile.write(str(instr) + ',' + str(prediction[instr].get('high')) + ',' + str(
                prediction[instr].get('low')) + ',' + str(
                prediction[instr].get('close')) + '\n')
        outfile.close()

    def data2sheet(self, write_raw=False, improve_model=False, maxdate=None,
                   complete=True, read_raw=False, append_raw=False):
        """
        This method will take the input collected from oanda and forexfactory and merge in a Data Frame
        write_raw: Write data frame used for training to disk
        read_raw: Read data frame used for training from disk
        write_predict: Write prediction file to disk to use it for trading later on
        improve_model: Perform Hyper parameter improvement for the estimators
        maxdate: Maximum data to use in the prediction. If None use all.
        new_estim: Build new estimators with new Hyper parameters
        """
        self.num_samples = 1
        raw_name = self.settings.get('raw_name')
        has_contributed = False
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
                statement = "select distinct date from dailycandles where date <= '" + maxdate + "' and " + c_cond + " order by date;"
            else:
                statement = "select distinct date from dailycandles where " + c_cond + " order by date;"
            if append_raw:
                df_prev = pd.read_csv(raw_name)
            else:
                df_prev = None
            for row in self.db.query(statement):
                # if row['date'][:4] == year:
                date = row['date']
                if append_raw:
                    if np.any(date == df_prev['date']):
                        continue
                date_split = date.split('-')
                weekday = int(datetime.datetime(int(date_split[0]), int(date_split[1]), int(date_split[2])).weekday())
                if weekday == 4 or weekday == 5:  # saturday starts on friday and sunday on saturday
                    continue
                dates.append(row['date'])
            # dates = dates[-100:] # use this line to decrease computation time for development
            bar = None
            if self.verbose > 0:
                print('INFO: Starting data frame preparation')
                bar = progressbar.ProgressBar(maxval=len(dates),
                                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()
            index = 0
            df_dict = []
            for date in dates:
                print(date)
                if self.verbose > 0:
                    bar.update(index)
                index += 1
                for i in range(self.num_samples):  # 2 fold over sampling
                    df_row = self.get_df_for_date(date, inst, complete, bootstrap=False)  # improve_model)
                    # df_row = merge_dicts(df_row, yest_df, '_yester')
                    if df_row:
                        has_contributed = True
                        df_dict.append(df_row)
            df = pd.DataFrame(df_dict)
            if append_raw:
                df = pd.concat([df_prev, df], axis=0)
            if self.verbose > 0:
                bar.finish()
        if write_raw:
            print('Constructed DF with shape ' + str(df.shape))
            if has_contributed:
                df.to_csv(raw_name, index=False)
        date_column = df['date'].copy()  # copy for usage in improveEstim
        df.drop(['date'], 1, inplace=True)
        bar = progressbar.ProgressBar(maxval=len(df.columns),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        if self.verbose > 0:
            print('INFO: Starting prediction')
            bar.start()
        index = 0
        importances = []
        mse_list = []
        mae_list = []
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
            if '_yester' in col:  # skip yesterday stuff for prediction
                continue
            col_instrument = '_'.join(parts[:2])
            # if not col_instrument in self.tradeable_instruments:
            #     continue
            if improve_model:
                this_importances, mse, mae = self.improve_estimator(col, df)
                mse_list.append(mse)
                mae_list.append(mae)
                for imp in this_importances:
                    importances.append({'estimator': col, 'label': imp['label'], 'importance': imp['importance']})
        if self.verbose > 0:
            bar.finish()
        df = pd.DataFrame(importances)
        df.to_csv('importance.csv', index=False)
        print('Final Mean Loss ' + str(np.mean(mse_list)) + ' / ' + str(np.mean(mae_list)))

    def save_prediction_to_db(self, date):
        prediction_df = pd.read_csv(self.settings['prices_path'])
        for index, row in prediction_df.iterrows():
            prediction_object = { 'instrument': row['INSTRUMENT'], 'date': date, 'high': row['HIGH'], 'low': row['LOW'], 'close': row['CLOSE'] }
            print('Saving to disk {prediction_object}'.format(prediction_object=str(prediction_object)))
            self.prediction_table.upsert(prediction_object, ['instrument', 'date'])
        

    def improve_estimator(self, col, df):
        estim = estimator.Estimator(col)
        return estim.improve_estimator(df, estimtable=self.estimtable, num_samples=self.num_samples,
                                        estimpath=self.settings.get('estim_path'))

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
                estim = estimator.Estimator(pcol, self.settings.get('estim_path'))
            except:
                print('Failed to load model for ' + pcol)
                continue
            if self.verbose > 1:
                print(pcol)
            for name, importance in zip(feature_names, estim.get_feature_importances()):
                feature_importance = {'name': pcol, 'feature': name, 'importance': importance}
                self.importances.upsert(feature_importance, ['name', 'feature'])

    @staticmethod
    def dist_to_now(input_date):
        now = datetime.datetime.now()
        ida = datetime.datetime.strptime(input_date, '%Y-%m-%d')
        delta = now - ida
        return math.exp(-delta.days / 365.25)  # exponentially decaying weight decay

    def load_keras(self, df):
        if not self.keras_model:
            self.keras_model = self.estimator_keras('', df)
        return self.keras_model

    def predict_column(self, predict_column, df):
        # Predict the next outcome for a given column
        # predict_column: Columns to predict
        # df: Data Frame containing the column itself as well as any features
        # new_estimator: Whether to lead the existing estimator from disk or create a new one

        try:
            estim = estimator.Estimator(predict_column, estimpath=self.settings.get('estim_path'))
        except:
            print('Could not load estimator for ' + str(predict_column))
            return None
        yp = estim.predict(df.iloc[-1, :])
        return yp

    def get_units(self, dist, ins):
        """
        get the number of units to trade for a given pair
        dist: Distance to the SL
        ins: Instrument to trade, e.g. EUR_USD
        """

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
        if dist:
            multiplier = min(price / dist, 100)  # no single trade can be larger than the account NAV
        else:
            multiplier = 100
        if not conversion:
            print('CRITICAL: Could not convert ' + leading_currency + '_' + trailing_currency + ' to EUR')
            return 0  # do not place a trade if conversion fails
        raw_units = multiplier * target_exposure * conversion
        if raw_units > 0:
            return math.floor(raw_units)
        else:
            return math.ceil(raw_units)

    def convert_units(self, units, ins):
        """
        get the number of units to trade for a given pair
        dist: Distance to the SL
        ins: Instrument to trade, e.g. EUR_USD
        """

        trailing_currency = ''
        leading_currency = ins.split('_')[0]
        price = self.get_price(ins)
        # each trade should risk 1% of NAV at SL at most. Usually it will range
        # around 0.1 % - 1 % depending on expectation value
        target_exposure = units
        conversion = self.get_conversion(leading_currency)
        if not conversion:
            trailing_currency = ins.split('_')[1]
            conversion = self.get_conversion(trailing_currency)
            if conversion:
                conversion = conversion / price
        if not conversion:
            print('CRITICAL: Could not convert ' + leading_currency + '_' + trailing_currency + ' to EUR')
            return 0  # do not place a trade if conversion fails
        raw_units = target_exposure * conversion
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

    def check_end_of_day(self):
        """
        check all open trades and check whether one of them would possible fall victim to spread widening
        """
        pl_list = dict()
        while len(self.trades) > 0:
            try:
                self.trades = self.oanda.trade.list_open(self.settings.get('account_id')).get('trades', '200')
                for trade in self.trades:
                    uPL = float(trade.unrealizedPL)
                    if trade.id in pl_list.keys():
                        print(trade.instrument + ' ' + str(uPL) + ' ' + str(pl_list[trade.id]))
                        if pl_list[trade.id] > uPL:
                            print('Closing trade ' + str(trade.instrument))
                            response = self.oanda.trade.close(self.settings.get('account_id'), trade.id)
                        else:
                            print('Keeping trade ' + str(trade.instrument))
                            # print(response.raw_body)
                    pl_list[trade.id] = uPL
            except Exception as e:
                print(str(e))
            time.sleep(60)

    def open_limit(self, ins, duration=8, use_keras=False, use_stoploss=True, close_only=False):
        """
        Open orders and close trades using the predicted market movements

        Parameters
        ------
        ins: Instrument to trade

        Keyword Arguments
        ------
        duration: For how many hours should the limit order by placed
        use_keras: Whether to use the prediction produced by keras
        use_stoploss: Whether to use a stop loss

        Returns
        ------
        None
        """

        try:
            rr_target = 2
            if use_keras:
                price_path = self.settings['keras_path']
            else:
                price_path = self.settings['prices_path']
            df = pd.read_csv(price_path)
            candles = self.get_candles(ins, 'D', 1)
            candle = candles[0]
            now_str = datetime.datetime.now().strftime('%Y-%m-%d')
            #print(ins + ' ' + now_str + ' ' + candle['time'][:10])
            #if candle['time'][:10] == now_str:
            ##if int(candle.get('complete')):
            #    op = float(candle.get('mid').get('o'))
            #else:
            op = float(candle.get('mid').get('c'))
            cl = df[df['INSTRUMENT'] == ins]['CLOSE'].values[0]
            hi = df[df['INSTRUMENT'] == ins]['HIGH'].values[0] + op
            lo = df[df['INSTRUMENT'] == ins]['LOW'].values[0] + op
            price = self.get_price(ins)
            if hi < price or lo > price:
                return
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
            spread = self.get_spread(ins, spread_type='current')
            bid, ask = self.get_bidask(ins)
            trades = []
            currentUnits = 0
            min_units = 0
            for tr in self.trades:
                if tr.instrument == ins:
                    #if float(tr.currentUnits) * cl < 0:
                    #    print('Attempting to close ' + tr.instrument)
                    #    response = self.oanda.trade.close(self.settings.get('account_id'), tr.id)
                    #else:
                    if min_units == 0:
                        min_units = int(tr.currentUnits)
                    elif abs(min_units) > int(tr.currentUnits):
                        min_units = int(tr.currentUnits)
                    currentUnits += float(tr.currentUnits)
                    print('Keeping ' + tr.instrument + ' ' + str(tr.currentUnits) + ' ' + str(cl))
                    if cl > 0:
                        tp = hi
                    else:
                        tp = lo
                    pip_location = self.get_pip_size(ins)
                    pip_size = 10 ** (-pip_location + 1)
                    format_string = '30.' + str(pip_location) + 'f'
                    tp = format(tp, format_string).strip()
                    if tr.takeProfitOrder:
                        response = self.oanda.order.take_profit_replace(self.settings.get('account_id'), tr.takeProfitOrder.id, **{'tradeID': tr.id, 'price': tp})
                    else:
                        response = self.oanda.order.take_profit(self.settings.get('account_id'), **{'tradeID': tr.id, 'price': tp})
                    print(response.raw_body)
                    trades.append(tr)
            if close_only and len(trades) == 0:
                return
            if use_stoploss or len(trades) == 0:
                # here we calculate am entry based on having 
                if len(trades) > 0:
                    return
                if abs(close_score) > 1:
                    return
                #if float(candle.get('mid').get('h')) > hi:
                #    return
                #if float(candle.get('mid').get('l')) < lo:
                #    return
            if len(trades) == 0:
                if cl > 0:
                    step = (hi - lo)/2
                    sl = lo - step - spread
                    entry = min(lo, bid)
                    sldist = step
                    tp = hi
                else:
                    step = (hi - lo)/2
                    sl = hi + step + spread
                    entry = max(hi, ask)
                    sldist = step
                    tp = lo
                #rr = abs((tp - entry) / sldist )
                #if rr < rr_target:  # Risk-reward too low
                #    if cl > 0:
                #        entry = sl + (tp - sl)/(rr_target+1.0)
                #        sldist = entry - sl + spread
                #    else:
                #        entry = sl - (sl - tp)/(rr_target+1.0)
                #        sldist = sl - entry + spread
                # if you made it here its fine, lets open a limit order
                units = math.floor(abs(self.get_units(sldist, ins) * min(abs(cl), 1.0)))
                if tp < sl:
                    units *= -1
                if units * currentUnits > 0:
                    if abs(units) > abs(currentUnits):
                        units -= currentUnits
                #else:
                #    units -= currentUnits
                if abs(units) < 1:
                    return None  # oops, risk threshold too small
                relative_cost = spread / sldist
                #print(ins + ' - ' + str(spread) + ' - ' + str(relative_cost) + ' - ' + str(tp) + ' ' + str(entry))
                #return None
                if 0.15 <= relative_cost:
                    return None  # edge too small to cover cost
            else:
                sl = 0
                if min_units > 0:
                    entry = lo
                    tp = hi
                else:
                    entry = hi
                    tp = lo
                sldist = abs(entry - tp)
                units = min_units
            pip_location = self.get_pip_size(ins)
            pip_size = 10 ** (-pip_location + 1)
            format_string = '30.' + str(pip_location) + 'f'
            tp = format(tp, format_string).strip()
            sl = format(sl, format_string).strip()
            sldist = format(sldist, format_string).strip()
            entry = format(entry, format_string).strip()
            expiry = datetime.datetime.now() + datetime.timedelta(hours=duration)
            if use_stoploss:
                args = {'order': {
                    'instrument': ins,
                    'units': units,
                    'price': entry,
                    'type': 'LIMIT',
                    'timeInForce': 'GTD',
                    'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'takeProfitOnFill': {'price': tp, 'timeInForce': 'GTC'},
                    # 'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'}
                    'trailingStopLossOnFill': {'distance': sldist, 'timeInForce': 'GTC'}
                }}
            else:
                args = {'order': {
                    'instrument': ins,
                    'units': units,
                    'price': entry,
                    'type': 'LIMIT',
                    'timeInForce': 'GTD',
                    'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'takeProfitOnFill': {'price': tp, 'timeInForce': 'GTC'}
                    # 'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'}
                    #'trailingStopLossOnFill': {'distance': sldist, 'timeInForce': 'GTC'}
                }}
            if self.write_trades:
                self.trades_file.write(str(ins) + ',' + str(units) + ',' + str(tp) + ',' + str(sl) + ',' + str(
                    entry) + ',' + expiry.strftime('%Y-%m-%dT%M:%M:%S.%fZ') + ';')
            #if self.verbose > 1:
            #    print(args)
            print(ins + ' - ' + str(cl) + ' - ' + str(units))
            ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
            #if self.verbose > 1:
            #    print(ticket.raw_body)
        except Exception as e:
            print('failed to open for ' + ins)
            print(e)

    def get_new_symbol(self):
        """
        This Method will scan the available symbols and return the one with best RR

        Returns
        ------
        string symbol that should be traded next
        """
        df = pd.read_csv(self.settings['prices_path'])
        ratios = []
        for ins in self.allowed_ins:
            candles = self.get_candles(ins.name, 'D', 1)
            candle = candles[0]
            op = float(candle.get('mid').get('c'))
            cl = df[df['INSTRUMENT'] == ins.name]['CLOSE'].values[0]
            hi = df[df['INSTRUMENT'] == ins.name]['HIGH'].values[0] + op
            lo = df[df['INSTRUMENT'] == ins.name]['LOW'].values[0] + op
            price = self.get_price(ins.name)
            if price < lo or price > hi:
                ratio = 0
            elif cl > 0:
                ratio = cl * (hi - price)/(price - lo)
            elif cl < 0:
                ratio = abs(cl) * (price - lo)/(hi - price)
            # print('{ins} - {ratio}'.format(ins=ins.displayName, ratio=str(ratio)))
            if ratio > 0:
                ratios.append({'ins': ins.name, 'ratio': ratio })
        if len(ratios) == 0:
            return None
        ratios = sorted(ratios, key=lambda x: x.get('ratio'), reverse=True)
        return ratios

    def get_margin_ratio(self):
        account = self.oanda.account.summary(self.settings.get('account_id')).get('account', '200')
        margin_used = float(account.marginUsed)
        margin_avail = float(account.marginAvailable)
        print('Relative Margin used {ratio}'.format(ratio=str(margin_used/margin_avail)))
        return margin_used / margin_avail

    def manage_portfolio(self, close_only=False):
        target_ratio = 0.1
        exposures = {}
        for trade in self.trades:
            leading_symbol, trailing_symbol = trade.instrument.split('_')
            if not leading_symbol in exposures.keys():
                exposures[leading_symbol] = 0
            if not trailing_symbol in exposures.keys():
                exposures[trailing_symbol] = 0
            price = self.get_price(trade.instrument)
            exposures[leading_symbol] += trade.currentUnits
            exposures[trailing_symbol] -= trade.currentUnits * price
            # self.manage_trade(trade)
        if close_only:
            return
        ratios = self.get_new_symbol()
        # try to enter the 5 best opps as long
        account = self.oanda.account.summary(self.settings.get('account_id')).get('account', '200')
        margin_used = float(account.marginUsed)
        margin_avail = float(account.marginAvailable)
        print('Relative Margin used {ratio}'.format(ratio=str(margin_used/margin_avail)))
        if margin_used / margin_avail > target_ratio:
            return
        for ratio in ratios:
            if ratio['ratio'] > 1.5:
                print('Opening new trade with {ins} / {ratio}'.format(ins=ratio.get('ins'), ratio=str(ratio.get('ratio'))))
                if self.simplified_trader(ratio['ins'], exposures):
                    break

    def manage_trade(self, trade):
        ins = trade.instrument
        candles = self.get_candles(ins, 'D', 1)
        candle = candles[0]
        op = float(candle.get('mid').get('c'))
        df = pd.read_csv(self.settings['prices_path'])
        cl = df[df['INSTRUMENT'] == ins]['CLOSE'].values[0]
        hi = df[df['INSTRUMENT'] == ins]['HIGH'].values[0] + op
        lo = df[df['INSTRUMENT'] == ins]['LOW'].values[0] + op
        price = self.get_price(ins)
        if cl * trade.currentUnits < 0:
            print('Closing Trade {ins}'.format(ins=trade.instrument))
            self.oanda.trade.close(self.settings.get('account_id'), trade.id)
            return
        if abs(trade.currentUnits) > 0.6*abs(trade.initialUnits):
            if cl > 0:
                daily_target = abs(cl) * (hi - op) + op
                if price > daily_target:
                    args = {'order': {
                        'instrument': trade.instrument,
                        'units': str(-int(math.ceil(float(trade.currentUnits)/2))),
                        'type': 'MARKET'
                    }}
                    ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
            elif cl < 0:
                daily_target = op - abs(cl) * (op - lo)
                if price < daily_target:
                    args = {'order': {
                        'instrument': trade.instrument,
                        'units': str(-int(trade.currentUnits/2)),
                        'type': 'MARKET'
                    }}
                    ticket = self.oanda.order.create(self.settings.get('account_id'), **args)

    def simplified_trader(self, ins, exposures):
        """
        Open orders and close trades using the predicted market movements
        close_only: Set to true to close only without checking for opening Orders
        complete: Whether to use only complete candles, which means to ignore the incomplete candle of today
        """

        try:
            min_ratio = 1.5
            df = pd.read_csv(self.settings['prices_path'])
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
                return False
            column_name = ins + '_high'
            high_score = self.get_score(column_name)
            if not high_score:
                return False
            column_name = ins + '_low'
            low_score = self.get_score(column_name)
            if not low_score:
                return False
            spread = self.get_spread(ins, spread_type='trading')
            bid, ask = self.get_bidask(ins)
            trades = []
            current_units = 0
            for tr in self.trades:
                if tr.instrument == ins:
                    trades.append(tr)
                    current_units += int(tr.currentUnits)
                    return False
            if abs(close_score) > 1:
                return False
            if cl > 0:
                sl = ask - ( hi - ask)/ min_ratio
                tp = hi
                ratio = (tp - price)/(price - sl)
            else:
                sl = bid + ( hi - bid )/min_ratio
                tp = lo
                ratio = (price - tp)/(sl - price)
            # if you made it here its fine, lets open a limit order
            # r2sum is used to scale down the units risked to accomodate the estimator quality
            units = self.get_units(None, ins) * min(abs(cl),
                                                              1.0) * (1 - abs(close_score))
            if tp < sl:
                units *= -1
            units -= current_units
            if abs(units) < 1:
                return False  # oops, risk threshold too small
            leading_symbol, trailing_symbol = ins.split('_')
            if leading_symbol in exposures.keys():
                if exposures[leading_symbol] * units > 0:
                    print('Cancel new trade since {symbol} exposure is already at {units} units'.format(symbol=leading_symbol, units=str(exposures[leading_symbol])))
                    return False
            if trailing_symbol in exposures.keys():
                if exposures[trailing_symbol] * units < 0:
                    print('Cancel new trade since {symbol} exposure is already at {units} units'.format(symbol=trailing_symbol, units=str(exposures[trailing_symbol])))
                    return False
            relative_cost = spread / abs(tp - op)
            if abs(cl) <= relative_cost:
                print('relative cost too large')
                return False  # edge too small to cover cost
            pip_location = self.get_pip_size(ins)
            pip_size = 10 ** (-pip_location + 1)
            # if abs(sl - entry) < 200 * 10 ** (-pip_location):  # sl too small
            #    return None
            # otype = 'MARKET'
            otype = 'MARKET'
            format_string = '30.' + str(pip_location) + 'f'
            tp = format(tp, format_string).strip()
            sl = format(sl, format_string).strip()
            units = str(int(units))
            args = {'order': {
                'instrument': ins,
                'units': units,
                'type': otype,
                'takeProfitOnFill': {'price': tp, 'timeInForce': 'GTC'}
                #'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'}
            }}
            #if self.write_trades:
            #    self.trades_file.write(str(ins) + ',' + str(units) + ',' + str(tp) + ',' + str(sl) + ',' + str(
            #        entry) + ',' + expiry.strftime('%Y-%m-%dT%M:%M:%S.%fZ') + ';')
            if self.verbose > 1:
                print(args)
            #print(ins + ' - ' + str(cl) + ' - ' + str(units))
            ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
            print(ticket.raw_body)
            return True
        except Exception as e:
            print('failed to open for ' + ins)
            print(e)


    def move_stops(self):
        """
        Move the stop of all winning trades to (price + entry)/2
        """
        for trade in self.trades:
            if trade.unrealizedPL > 0:
                price = self.get_price(trade.instrument)
                entry = trade.price
                newSL = (price+entry)/2
                currentSL = trade.stopLossOrder.price
                if trade.currentUnits > 0 and newSL > currentSL:
                    pip_location = self.get_pip_size(trade.instrument)
                    pip_size = 10 ** (-pip_location + 1)
                    format_string = '30.' + str(pip_location) + 'f'
                    newSL = format(newSL, format_string).strip()
                    args = { 'order': {
                            'tradeID': trade.id,
                            'price': newSL,
                            'type': 'STOP_LOSS'
                        }}
                    response = self.oanda.order.cancel(self.settings.get('account_id'), trade.stopLossOrder.id)
                    response = self.oanda.order.create(self.settings.get('account_id'), **args)
                    print(response.raw_body)

    def simple_limits(self, ins, duration=20):
        """
        Open orders and close trades using the predicted market movements

        Parameters
        ------
        ins: Instrument to trade

        Keyword Arguments
        ------
        duration: For how many hours should the limit order by placed
        use_keras: Whether to use the prediction produced by keras
        use_stoploss: Whether to use a stop loss

        Returns
        ------
        None
        """

        try:
            current_direction = 0
            for trade in self.trades:
                if trade.instrument == ins:
                    current_direction += float(trade.currentUnits)
            price_path = self.settings['prices_path']
            df = pd.read_csv(price_path)
            candles = self.get_candles(ins, 'D', 1)
            candle = candles[0]
            spread = self.get_spread(ins, spread_type='trading')
            op = float(candle.get('mid').get('o'))
            cl = df[df['INSTRUMENT'] == ins]['CLOSE'].values[0]
            hi = df[df['INSTRUMENT'] == ins]['HIGH'].values[0] + op
            lo = df[df['INSTRUMENT'] == ins]['LOW'].values[0] + op
            sl_lo = lo - ( hi - lo )
            sl_hi = hi + ( hi - lo )
            if current_direction > 0:
                hi -= spread
            if current_direction < 0:
                lo += spread
            units = self.convert_units(self.settings.get('account_risk'), ins) # math.floor(abs(self.get_units(sldist, ins) * min(abs(cl), 1.0)))
            pip_location = self.get_pip_size(ins)
            pip_size = 10 ** (-pip_location + 1)
            format_string = '30.' + str(pip_location) + 'f'
            lo = format(lo, format_string).strip()
            sl_lo = format(sl_lo, format_string).strip()
            hi = format(hi, format_string).strip()
            sl_hi = format(sl_hi, format_string).strip()
            expiry = datetime.datetime.now() + datetime.timedelta(hours=duration)
            print(ins + ' - ' + str(cl) + ' - ' + str(units))
            if cl > 0:
                _units = int(units*abs(cl))
                args = {'order': {
                    'instrument': ins,
                    'units': _units,
                    'price': lo,
                    'type': 'LIMIT',
                    'timeInForce': 'GTD',
                    'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'takeProfitOnFill': {'price': hi, 'timeInForce': 'GTC'},
                    'stopLossOnFill': {'price': sl_lo, 'timeInForce': 'GTC'}
                    # 'trailingStopLossOnFill': {'distance': sldist, 'timeInForce': 'GTC'}
                }}
                # if current_direction <= units:
                ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
                print(ticket.raw_body)
            if cl < 0:
                _units = -int(units*abs(cl))
                args = {'order': {
                    'instrument': ins,
                    'units': _units,
                    'price': hi,
                    'type': 'LIMIT',
                    'timeInForce': 'GTD',
                    'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                    'takeProfitOnFill': {'price': lo, 'timeInForce': 'GTC'},
                    'stopLossOnFill': {'price': sl_hi, 'timeInForce': 'GTC'}
                    # 'trailingStopLossOnFill': {'distance': sldist, 'timeInForce': 'GTC'}
                }}
                ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
                print(ticket.raw_body)
            #if self.verbose > 1:
            #    print(ticket.raw_body)
        except Exception as e:
            print('failed to open for ' + ins)
            print(e)
