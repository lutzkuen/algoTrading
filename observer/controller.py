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
import dataset


class controller(object):

    def __init__(self, confname):
        config = configparser.ConfigParser()
        config.read(confname)
        self.settings = {}
        self.settings['domain'] = config.get('live',
                'streaming_hostname')
        self.settings['access_token'] = config.get('live', 'token')
        self.settings['account_id'] = config.get('live',
                'active_account')
        self.settings['v20_host'] = config.get('live', 'hostname')
        self.settings['v20_port'] = config.get('live', 'port')
        self.oanda = v20.Context(self.settings.get('v20_host'),
                                 port=self.settings.get('v20_port'),
                                 token=self.settings.get('access_token'
                                 ))
        self.allowed_ins = \
            self.oanda.account.instruments(self.settings.get('account_id'
                )).get('instruments', '200')
        self.db = dataset.connect('sqlite:////home/ubuntu/data/barsave.db')
        self.table = self.db['dailycandles']
    def retrieveData(self, numCandles):
        for ins in self.allowed_ins:
            candles = self.getCandles(ins.name,'D',numCandles)
            self.candlesToDB(candles, ins.name)
    def candlesToDB(self, candles, ins):
        for candle in candles:
         if not bool(candle.get('complete')):
          continue
         time = candle.get('time')[:10]# take the YYYY-MM-DD part
         icandle = self.table.find_one(date = time, ins = ins)
         if icandle:
          print(ins + ' ' + time + ' already in dataset')
          continue
         cobj = { 'ins': ins, 'date': time, 'open': candle.get('mid').get('o'), 'close': candle.get('mid').get('c'), 'high': candle.get('mid').get('h'), 'low': candle.get('mid').get('l'), 'volume': candle.get('volume') }
         print('Inserting ' + str(cobj))
         self.table.insert(cobj)
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
        #print(response.raw_body)
        candles = json.loads(response.raw_body)
        return candles.get('candles')
