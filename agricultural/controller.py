#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import code
try:
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
        self.optimization_db = dataset.connect(config.get('data', 'optimization_path'))
        self.calendar = self.calendar_db['calendar']
        self.table = self.db['dailycandles']
        self.estimtable = self.db['estimators']
        self.importances = self.db['feature_importances']
        self.opt_table = self.optimization_db['function_values']
        # the following arrays are used to collect aggregate information in estimator improvement
        self.accuracy_array = []
        self.n_components_array = []
        self.spreads = {}
        self.prices = {}

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

    def get_spread(self, ins):
        # Returns spread for a instrument
        # ins: Instrument, e.g. EUR_USD

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
                print(str(e))
                time.sleep(1)
        price = json.loads(price_raw.raw_body)
        spread = abs(float(price.get('prices')[0].get('bids')[0].get('price'
                                                                     )) - float(price.get('prices')[0].get('asks'
                                                                                                           )[0].get(
            'price')))
        self.spreads[ins] = spread
        return spread

    def get_price(self, ins):
        # Returns price for a instrument
        # ins: Instrument, e.g. EUR_USD

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

    def get_relative_change(self, bar):
        return (float(bar.get('mid').get('c')) - float(bar.get('mid').get('o')))/float(bar.get('mid').get('o'))

    def open_order(self, ins, side):
        for order in self.orders:
            try:
                if order.instrument == ins:
                    # cancel the order
                    self.oanda.order.cancel(self.settings.get('account_id'), order.id)
            except Exception as e:
                pass
        current_position = sum([trade.currentUnits for trade in self.trades if trade.instrument == ins])
        mid_price = self.get_price(ins)
        expiry = datetime.datetime.now() + datetime.timedelta(hours=24)
        units = int((self.get_sign(side) * self.settings.get('account_risk')/mid_price) - current_position)
        if units == 0:
            return
        pip_location = self.get_pip_size(ins)
        format_string = '30.' + str (pip_location) + 'f'
        mid_price = format(mid_price, format_string).strip()
        args = {'order': {
                'instrument': ins,
                'units': units,
                #'price': mid_price,
                'type': 'MARKET'
                #'timeInForce': 'GTD',
                #'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                 #'trailingStopLossOnFill': {'distance': sldist, 'timeInForce': 'GTC'}
            }}
        resp = self.oanda.order.create(self.settings.get('account_id'), **args)
        print(resp.raw_body)
    def get_sign(self, num):
        if num == 0:
           return 0
        return int(num/abs(num))

    def set_position(self, sign_corn, sign_wheat, sign_soybn):
        print('set Position CORN: ' + str(sign_corn) + ' WHEAT: ' + str(sign_wheat) + ' SOYBN: ' + str(sign_soybn))
        self.open_order('CORN_USD', sign_corn)
        self.open_order('WHEAT_USD', sign_wheat)
        self.open_order('SOYBN_USD', sign_soybn)
        return 'great success'
    

    def open_orders(self):
        # get the last complete bar for CORN, WHEAT, SOYBN
        bar_corn = [candle for candle in self.get_candles('CORN_USD', 'D', 2) if candle.get('complete') == True][-1]
        bar_wheat = [candle for candle in self.get_candles('WHEAT_USD', 'D', 2) if candle.get('complete') == True][-1]
        bar_soybn = [candle for candle in self.get_candles('SOYBN_USD', 'D', 2) if candle.get('complete') == True][-1]
        change_corn = self.get_relative_change(bar_corn)
        change_wheat = self.get_relative_change(bar_wheat)
        change_soybn = self.get_relative_change(bar_soybn)
        print('CORN: ' + str(change_corn) + '\n' + 'WHEAT: ' + str(change_wheat) + '\n' + 'SOYBN: ' + str(change_soybn))
        if change_corn*change_wheat < 0:
            if change_corn*change_soybn < 0:
                return self.set_position(-change_corn, 0, 0)
            elif change_wheat*change_soybn < 0:
                return self.set_position(0, -change_wheat, 0)
        elif change_corn*change_soybn < 0:
            return self.set_position(0, 0, -change_soybn)
        set_corn = 0
        set_wheat = 0
        set_soybn = 0
        if change_corn > max([change_wheat, change_soybn]):
            set_corn = -1
        elif change_wheat > max([change_corn, change_soybn]):
            set_wheat = -1
        elif change_soybn > max([change_corn, change_wheat]):
            set_soybn = -1
        if change_corn < min([change_wheat, change_soybn]):
            set_corn = 1
        elif change_wheat < min([change_corn, change_soybn]):
            set_wheat = 1
        elif change_soybn < min([change_corn, change_wheat]):
            set_soybn = 1
        self.set_position(set_corn, set_wheat, set_soybn)
