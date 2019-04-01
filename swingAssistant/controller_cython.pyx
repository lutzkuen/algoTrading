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
            else:
                args = { 'order': {
                    'instrument': ins,
                    'units': units,
                    'type': 'MARKET' } }
                response = self.oanda.order.create(self.settings.get('account_id'), **args)
                print(response.raw_body)
