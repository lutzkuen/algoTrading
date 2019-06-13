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

import numpy as np
import pandas as pd

try:
    # noinspection PyUnresolvedReferences
    import v20
    # noinspection PyUnresolvedReferences
    from v20.request import Request
    v20present = True
except ImportError:
    print('WARNING: V20 library not present. Connection to broker not possible')
    v20present = False


class Controller(object):
    """
    This class controls most of the program flow for:
    - getting the data
    - constructing the data frame for training and prediction
    - actual training and prediction of required models
    - acting in the market based on the prediction
    """

    def __init__(self, config_name, _type, verbose=2):
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

        self.settings = {}

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
        # the following arrays are used to collect aggregate information in estimator improvement
        self.spreads = {}
        self.prices = {}

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

    def open_limit(self, row):
        """
        """
        ins = row['SYMBOL']
        hi = row ['UPPER']
        lo = row['LOWER']
        units = row['UNITS']
        sl = row['SL']
        tp = row['TP']
        dtend = datetime.datetime.strptime(row['DTEND'], '%Y-%m-%d')
        if datetime.datetime.now() > dtend:
            print(' Obsolete line in orders: ' + str(row))
            return
        price = self.get_price(ins)
        bid, ask = self.get_bidask(ins)
        for tr in self.trades:
            try:
                if tr.instrument == ins:
                    if ( abs(tr.currentUnits) >= abs(units) ) or ( tr.currentUnits * units < 0):
                        return
            except:
                pass
        for tr in self.orders:
            try:
                if tr.instrument == ins:
                    if ( abs(tr.units) >= abs(units) ) or ( tr.units * units < 0 ):
                        return
            except:
                pass
        if units < 0:
            if price < hi:
                otype = 'LIMIT'
                entry = lo
            else:
                otpye = 'STOP'
                entry = hi
        else:
            if price > lo:
                otype = 'LIMIT'
                entry = hi
            else:
                otype = 'STOP'
                entry = lo
        pip_location = self.get_pip_size(ins)
        pip_size = 10 ** (-pip_location + 1)
        format_string = '30.' + str(pip_location) + 'f'
        tp = format(tp, format_string).strip()
        sl = format(sl, format_string).strip()
        entry = format(entry, format_string).strip()
        expiry = dtend
        args = {'order': {
            'instrument': ins,
            'units': units,
            'price': entry,
            'type': otype,
            'timeInForce': 'GTD',
            'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'takeProfitOnFill': {'price': tp, 'timeInForce': 'GTC'},
            'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'}
        }}
        ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
        print(ticket.raw_body)
