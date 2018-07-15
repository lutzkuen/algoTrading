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
Three Ducks Trend Following as found on BabyPips
https://forums.babypips.com/t/the-3-ducks-trading-system/6430
Use at own risk
Author: Lutz Kuenneke, 12.07.2018
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


class controller(object):

    def __init__(self, confname):
        config = configparser.ConfigParser()
        config.read(confname)
        self.settings = {}
        self.ducks = [{'name': 'duck1', 'granularity': 'H4',
                      'numCandles': 60}, {'name': 'duck2',
                      'granularity': 'H1', 'numCandles': 60},
                      {'name': 'duck3', 'granularity': 'M5',
                      'numCandles': 60}]
        self.settings['domain'] = config.get('demo',
                'streaming_hostname')
        self.settings['access_token'] = config.get('demo', 'token')
        self.settings['account_id'] = config.get('demo',
                'active_account')
        self.settings['v20_host'] = config.get('demo', 'hostname')
        self.settings['v20_port'] = config.get('demo', 'port')
        self.settings['stopLoss'] = int(config.get('ducks', 'stopLoss'))
        self.settings['takeProfit'] = int(config.get('ducks',
                'takeProfit'))
        self.settings['units'] = int(config.get('ducks', 'units'))
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

    def getSMA(
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
        candles = json.loads(response.raw_body)
        sma = np.mean([np.float(candle.get('mid').get('c'))
                      for candle in candles.get('candles')])  # sma over all returned candles
        return sma

    def checkIns(self, ins):
        price = self.getPrice(ins)

        # get the ducks

        smas = [self.getSMA(ins, duck.get('granularity'),
                duck.get('numCandles')) for duck in self.ducks]

        # first ducks must be aligned, third one in the making for us to place an order

        bullish = [sma < price for sma in smas] == [True, True, False]
        bearish = [sma > price for sma in smas] == [True, True, False]
        if bullish:
            print(ins + ' is bullish')
            direction = 1
        if bearish:
            print(ins + ' is bearish')
            direction = -1
        if not bullish and not bearish:
            print(ins + ' no clear signal, dont touch')
            return

        # check whether there is an open trade

        if len([trade for trade in self.trades if trade.instrument
               == ins and trade.currentUnits * direction > 0]) > 0:
            print('Skipping ' + ins)
            return
        pipLoc = self.getPipSize(ins)
        pipVal = 10 ** (-pipLoc + 1)
        entry = smas[2]  # use third ducks sma as entr
        tp = entry + direction * pipVal * self.settings.get('takeProfit'
                )
        sl = entry - direction * pipVal * self.settings.get('stopLoss')
        units = direction * self.settings.get('units')

        # round off for the v20 api to accept the stops

        fstr = '30.' + str(pipLoc) + 'f'
        tp = format(tp, fstr).strip()
        sl = format(sl, fstr).strip()
        entry = format(entry, fstr).strip()
        expiry = datetime.datetime.now() + datetime.timedelta(hours=1)
        args = {'order': {
            'instrument': ins,
            'units': units,
            'price': entry,
            'type': 'STOP',
            'timeInForce': 'GTD',
            'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'takeProfitOnFill': {'price': tp, 'timeInForce': 'GTC'},
            'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'},
            }}
        ticket = self.oanda.order.create(self.settings.get('account_id'
                ), **args)
        ticket_json = json.loads(ticket.raw_body)


        print(ticket_json)
