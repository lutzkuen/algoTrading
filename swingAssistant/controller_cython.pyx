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
import progressbar

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
        self.prices = dict()

        self.write_trades = write_trades
        if write_trades:
            trades_path = config.get('data', 'trades_path')
            self.trades_file = open(trades_path, 'w')
        self.settings = dict()
        if _type and v20present:
            self.settings['domain'] = config.get(_type,
                                                 'streaming_hostname')
            self.settings['access_token'] = config.get(_type, 'token')
            self.settings['account_id'] = config.get(_type,
                                                     'active_account')
            self.settings['v20_host'] = config.get(_type, 'hostname')
            self.settings['v20_port'] = config.get(_type, 'port')
            self.oanda = v20.Context(self.settings.get('v20_host'),
                                     port=self.settings.get('v20_port'),
                                     token=self.settings.get('access_token'
                                                             ))
            self.trades = self.oanda.trade.list_open(self.settings.get('account_id')).get('trades', '200')
            self.orders = self.oanda.order.list(self.settings.get('account_id')).get('orders', '200')

    def get_price(self, ins):
        """
        Returns price for a instrument
        ins: Instrument, e.g. EUR_USD
        """

        try:
            args = {'instruments': ins}
            if ins in self.prices.keys():
                return self.prices[ins]
            price_raw = self.oanda.pricing.get(self.settings.get('account_id'), **args)
            price_json = json.loads(price_raw.raw_body)
            price = (float(price_json.get('prices')[0].get('bids')[0].get('price')) + float(price_json.get('prices')[0].get('asks')[0].get('price'))) / 2.0
            self.prices[ins] = price
            return price
        except Exception as e:
            print(ins + 'get price ' + str(e))
            return None

    def reduce_exposure(self):
        # first thing we do is look for offsetting positions
        self.trades = self.oanda.trade.list_open(self.settings.get('account_id')).get('trades', '200')
        exposures = dict()
        total_exposures = dict()
        for trade in self.trades:
            price = self.get_price(trade.instrument)
            units = int(trade.currentUnits)
            curr_arr = trade.instrument.split('_')
            leading = curr_arr[0]
            trailing = curr_arr[1]
            if not leading in exposures.keys():
                total_exposures[leading] = 0
                exposures[leading] = []
            total_exposures[leading] += abs(units)
            exposures[leading].append({'units': units, 'instrument': trade.instrument, 'currency': leading, 'counter': trailing, 'counter_units': -units*price })
            if not trailing in exposures.keys():
                total_exposures[trailing] = 0
                exposures[trailing] = []
            total_exposures[trailing] += abs(units*price)
            exposures[trailing].append({'units': -units*price, 'instrument': trade.instrument, 'currency': trailing, 'counter': leading, 'counter_units': units })

        t_exposures = [{'currency': key, 'exposure': total_exposures[key]} for key in total_exposures.keys()]
        t_exposures = sorted(t_exposures, key = lambda x: x.get('exposure'), reverse = True)
        for exposure in t_exposures:
            ins = exposure.get('currency')
            for trade1 in exposures[ins]:
                for trade2 in exposures[trade1.get('counter')]:
                    if trade2['counter'] == ins:
                        continue
                    for trade3 in exposures[trade2.get('counter')]:
                        if trade1['counter_units']*trade2['units'] < 0 and trade2['counter_units']*trade3['units'] < 0 and trade3['counter'] == ins:
                            print('Identified ' + trade1['instrument'] + ' ' + trade2['instrument'] + ' ' + trade3['instrument'])
                            # instrument 1
                            ins1 = trade1['instrument']
                            if trade1['instrument'].split('_')[0] == trade1['currency']:
                                units1 = -trade1['units']
                            else:
                                units1 = -trade1['counter_units']
                            print('units1 ' + str(units1))
                            # code.interact(banner='', local=locals())
                            # instrument 2
                            ins2 = trade2['instrument']
                            if abs(trade2['units']) >= abs(trade1['counter_units']):
                                correction = abs(trade1['counter_units'])/abs(trade2['units'])
                                if trade2['instrument'].split('_')[0] == trade2['currency']:
                                    units2 = -trade2['units']
                                else:
                                    units2 = -trade2['counter_units']
                                units2 *= correction
                            else:
                                correction1 = abs(trade2['units']) / abs(trade1['counter_units'])
                                print('correction1: ' + str(correction1))
                                units1 *= correction1
                                if trade2['instrument'].split('_')[0] == trade2['currency']:
                                    units2 = -trade2['units']
                                else:
                                    units2 = -trade2['counter_units']

                            # instrument 3
                            ins3 = trade3['instrument']
                            if abs(trade3['units']) >= abs(trade2['counter_units']):
                                correction = abs(trade2['counter_units'])/abs(trade3['units'])
                                if trade3['instrument'].split('_')[0] == trade3['currency']:
                                    units3 = -trade3['units']
                                else:
                                    units3 = -trade3['counter_units']
                                units3 *= correction
                            else:
                                correction2 = abs(trade3['units']) / abs(trade2['counter_units'])
                                print('correction2: ' + str(correction2))
                                units1 *= correction2
                                units2 *= correction2
                                if trade3['instrument'].split('_')[0] == trade3['currency']:
                                    units3 = -trade3['units']
                                else:
                                    units3 = -trade3['counter_units']
                            for ins, raw_units in zip([ins1, ins2, ins3], [units1, units2, units3]):
                                units = str(int(raw_units))
                                args = { 'order': {
                                    'instrument': ins,
                                    'units': units,
                                    'type': 'MARKET' } }
                                ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
                                print(ticket.raw_body)
                                # print(args)
                            self.reduce_exposure()
                            return
        # if after closing offsetting positions the exposure is still too large we resort to closing the worst looser if that can be offset by winning trades
        print('Could not find circle trades, going to close the worst')
        account = self.oanda.account.summary(self.settings.get('account_id')).get('account', '200')
        if 10*float(account.balance) > float(account.positionValue):
            print('Account Balance ' + str(account.balance) + ' > ' + str(account.positionValue))
            return
        biggest_loss = 0
        loss_units = None
        loss_ins = None
        winning_trades = []
        total_win = 0
        for trade in self.trades:
            if float(trade.unrealizedPL) < biggest_loss:
                biggest_loss = float(trade.unrealizedPL)
                loss_units = int(trade.currentUnits)
                loss_ins = trade.instrument
            elif float(trade.unrealizedPL) > 0:
                winning_trades.append(trade)
                total_win += float(trade.unrealizedPL)
        winning_trades = sorted(winning_trades, key = lambda x: float(x.unrealizedPL))
        close_amount = min(abs(biggest_loss), total_win)
        close_units = -int(loss_units*close_amount/abs(biggest_loss))
        closed_amount = 0
        for trade in winning_trades:
            closed_amount += float(trade.unrealizedPL)
            self.oanda.trade.close(self.settings.get('account_id'), trade.id)
            print('Closing ' + trade.instrument)
            if closed_amount >= close_amount:
                break
        if loss_ins and abs(close_units) > 0:
            args = { 'order': {
                'instrument': loss_ins,
                'units': close_units,
                'type': 'MARKET' } }
            print(args)
            ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
            self.reduce_exposure()


    def check_positions(self):
        instruments = np.unique([trade.instrument for trade in self.trades])
        for ins in instruments:
            uPL = 0
            biggest_loss = 0
            winning_ids = []
            loss_id = None
            units = 0
            for trade in [trade for trade in self.trades if trade.instrument == ins]:
                if float(trade.unrealizedPL) < biggest_loss:
                    biggest_loss = float(trade.unrealizedPL)
                    loss_id = trade.id
                elif float(trade.unrealizedPL) > 0:
                    uPL += float(trade.unrealizedPL)
                    winning_ids.append(trade.id)
                if abs(trade.currentUnits) > abs(units):
                    units = trade.currentUnits
            if not loss_id:
                # let the winner run
                continue
            print(ins + ' ' + str(uPL) + ' ' + str(biggest_loss))
            if abs(uPL) > abs(biggest_loss):
                self.oanda.trade.close(self.settings.get('account_id'), loss_id)
                for trade_id in winning_ids:
                    self.oanda.trade.close(self.settings.get('account_id'), trade_id)
            # else:
            #     args = { 'order': {
            #         'instrument': ins,
            #         'units': units,
            #         'type': 'MARKET' } }
            #     response = self.oanda.order.create(self.settings.get('account_id'), **args)
            #     print(response.raw_body)
