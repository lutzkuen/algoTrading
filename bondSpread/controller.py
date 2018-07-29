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
Spread trading using UK/US 10yr bond yields
Use at own risk
Author: Lutz Kuenneke, 26.07.2018
"""
import json
import time
import v20
from v20.request import Request
import requests
import configparser
import math
import datetime
import numpy as np
import pandas as pd


class controller(object):

    def __init__(self, confname):
        config = configparser.ConfigParser()
        config.read(confname)
        self.settings = {}
        self.settings['domain'] = config.get('demo',
                'streaming_hostname')
        self.settings['access_token'] = config.get('demo', 'token')
        self.settings['account_id'] = config.get('demo',
                'active_account')
        self.settings['v20_host'] = config.get('demo', 'hostname')
        self.settings['v20_port'] = config.get('demo', 'port')
        self.oanda = v20.Context(self.settings.get('v20_host'),
                                 port=self.settings.get('v20_port'),
                                 token=self.settings.get('access_token'
                                 ))
        self.allowed_ins = \
            self.oanda.account.instruments(self.settings.get('account_id'
                )).get('instruments', '200')
        self.trades = \
            self.oanda.trade.list_open(self.settings.get('account_id'
                )).get('trades', '200')
        self.dataObj = {
        'uk10y': { 'name': 'UK10YB_GBP' },
        'us10y': { 'name':  'USB10Y_USD'},
        #'uk100': { 'name':  'UK100GBP'},
        #'us30': { 'name': 'US30_USD'},
        'gu': { 'name': 'GBP_USD'}}
    def retrieveData(self):
        for key in self.dataObj.keys():
            self.dataObj[key]['candles'] = self.getCandles(self.dataObj[key].get('name'),'H4',500)
    def writeData(self, fname):
        writeObj = {}
        for key in self.dataObj.keys():
            colname = self.dataObj[key].get('name')
            writeObj[colname] = [float(candle.get('mid').get('c')) for candle in self.dataObj[key]['candles']]
        df = pd.DataFrame.from_dict(writeObj)
        df.to_csv(fname)
    def getConversion(self, leadingCurr):
        # get conversion rate to account currency
        accountCurr = 'EUR'
        # trivial case
        if leadingCurr == accountCurr:
            return 1
        # try direct conversion
        for ins in self.allowed_ins:
            if leadingCurr in ins.name and accountCurr in ins.name:
                price = self.getPrice(ins.name)
                if ins.name.split('_')[0] == accountCurr:
                    return price
                else:
                    return 1.0 / price
        # try conversion via usd
        eurusd = self.getPrice('EUR_USD')
        for ins in self.allowed_ins:
            if leadingCurr in ins.name and 'USD' in ins.name:
                price = self.getPrice(ins.name)
                if ins.name.split('_')[0] == 'USD':
                    return price / eurusd
                else:
                    return 1.0 / (price * eurusd)
        print('CRITICAL: Could not convert ' + leadingCurr + ' to EUR')
        return None
    def getUnits(self, dist, ins):
        # get the number of units to trade for a given pair
        if dist == 0:
            return 0
        leadingCurr = ins.split('_')[0]
        price = self.getPrice(ins)
        # each trade should risk 1% of NAV at SL at most. Usually it will range
        # around 0.1 % - 1 % depending on expectation value
        targetExp = self.settings.get('account_risk')*0.01
        conversion = self.getConversion(leadingCurr)
        multiplier = min(price / dist, 100) # no single trade can be larger than the account NAV
        if not conversion:
            return 0  # do not place a trade if conversion fails
        return math.floor(multiplier * targetExp * conversion )

    def getPipSize(self, ins):
        pipLoc = [_ins.pipLocation for _ins in self.allowed_ins
                  if _ins.name == ins]
        if not len(pipLoc) == 1:
            return None
        return -pipLoc[0] + 1

    def getPrice(self, ins):
        args = {'instruments': ins}
        nprice = self.oanda.pricing.get(self.settings.get('account_id'
                ), **args)
        pobj = json.loads(nprice.raw_body)
        price = (float(pobj.get('prices')[0].get('bids')[0].get('price'
                 )) + float(pobj.get('prices')[0].get('asks'
                 )[0].get('price'))) / 2.0
        return price

    def getCandles(
        self,
        ins,
        granularity,
        numCandles,
        ):
        request = Request('GET',
                          '/v3/instruments/{instrument}/candles?count={count}&price={price}&granularity={granularity}'
                          )
        request.set_path_param('instrument', ins)
        request.set_path_param('count', numCandles)
        request.set_path_param('price', 'M')
        request.set_path_param('granularity', granularity)
        response = self.oanda.request(request)
        print(response.raw_body)
        candles = json.loads(response.raw_body)
        return candles.get('candles')
