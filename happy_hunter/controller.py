#!/usr/bin/env python
__author__ = "Lutz Künneke"
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Lutz Künneke"
__email__ = "lutz.kuenneke89@gmail.com"
__status__ = "Prototype"
"""
Happy Hunter price action trading system as described here
https://www.babypips.com/trading/happy-hunter-price-action-trading-system-fixedtp-v4-20180420
Use at own risk
This is the fixed TP Variant as it had better results when i last checked. All credits for the system go to the original contributirs on BabyPips
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
        self.settings['domain'] = config.get("demo", "streaming_hostname")
        self.settings['access_token'] = config.get("demo", "token")
        self.settings['account_id'] = config.get("demo", "active_account")
        self.settings['v20_host'] = config.get('demo', 'hostname')
        self.settings['v20_port'] = config.get('demo', 'port')
        self.settings['units'] = int(config.get('ducks', 'units'))
        self.oanda = v20.Context(self.settings.get('v20_host'),
                                 port=self.settings.get('v20_port'),
                                 token=self.settings.get('access_token'))
        self.allowed_ins = self.oanda.account.instruments(
            self.settings.get('account_id')).get('instruments', '200')
        self.trades = self.oanda.trade.list_open(
            self.settings.get('account_id')).get('trades', '200')
	    
    def getPipSize(self,ins):
        pipLoc = [_ins.pipLocation for _ins in self.allowed_ins if _ins.name == ins]
        if not len(pipLoc) == 1:
            return None
        return -pipLoc[0] + 1
    def getPrice(self, ins):
        args = {'instruments': ins}
        nprice = self.oanda.pricing.get(
            self.settings.get('account_id'), **args)
        pobj = json.loads(nprice.raw_body)
        price = (float(pobj.get('prices')[0].get('bids')[0].get(
            'price')) + float(pobj.get('prices')[0].get('asks')[0].get('price'))) / 2.0
        return price

    def getATRH(self, ins, granularity, numCandles):
        request = Request(
            'GET',
            '/v3/instruments/{instrument}/candles?count={count}&price={price}&granularity={granularity}')
        request.set_path_param('instrument', ins)
        request.set_path_param('count', numCandles)
        request.set_path_param('price', 'M')
        request.set_path_param('granularity', granularity)
        response = self.oanda.request(request)
        candles = json.loads(response.raw_body)
		self.candles = candles # will be used to check the other stuff
		self.sbar = self.candles[0]
        atr = np.mean([np.float(candle.get('mid').get('h'))-np.float(candle.get('mid').get('l')) for candle in candles.get('candles')]) # average over all TR's
        return atr
    def checkLHLL(self):
	    return np.float(self.candles[0].get('h')) < np.float(self.candles[1].get('h')) and np.float(self.candles[0].get('l')) < np.float(self.candles[1].get('l'))
    def checkHHHL(self):
	    return np.float(self.candles[0].get('h')) > np.float(self.candles[1].get('h')) and np.float(self.candles[0].get('l')) > np.float(self.candles[1].get('l'))
    
    def checkIns(self, ins):
	    self.atr = self.getATRH(ins)
        price = self.getPrice(ins)
        # get the ducks

		#If an LHLL Patterns Forms
	    lhll = self.checkLHLL():
		hhhl = self.checkHHHL()
		atrcond = np.float(self.candles[1].get('h')) - np.float(self.sbar.get('l')) > 1.5*self.atr
        #And if not all LOWS of the 12 candles prior to the SBar are higher than SBar LOW
        low12 = all(np.float(self.sbar.get('l')) < np.float(candle.get('l')) for candle in self.candles[1:13]]):
        #And if all LOWS of the 4 candles prior to the SBar are higher than SBar LOW
		low4 = all(np.float(self.sbar.get('l')) < np.float(candle.get('l')) for candle in self.candles[1:5]]):
        #And if (Bar2 Low > Bar1 Low > Sbar Low) is TRUE
		monlow2 = ( np.float(self.candles[2].get('l')) > np.float(self.candles[1].get('l')) and np.float(self.candles[1].get('l')) > np.float(self.sbar.get('l'))):
        #And if (Bar2 High > Bar1 High > Sbar High) is TRUE
		monhigh2 = ( np.float(self.candles[2].get('h')) > np.float(self.candles[1].get('h')) and np.float(self.candles[1].get('h')) > np.float(self.sbar.get('h'))):
		atrcond = np.float(self.candles[1].get('h')) - np.float(self.sbar.get('l')) > 1.5*self.atr:
		condition_triggered = []
		pipLoc = self.getPipSize(ins)
        pipVal = 10**(-pipLoc+1)
		spread = self.getSpread(ins)
		if [lhll, low12, atrcond ] == [True, True, False]: # A1
		    condition_triggered.append('A1')
			entry = np.float(self.candles[1].get('h')) + pipVal + spread
            tp = entry + direction * pipVal * self.settings.get('takeProfit')
            sl = entry - direction * pipVal * self.settings.get('stopLoss')
            units = self.settings.get('units')
            # round off for the v20 api to accept the stops
            fstr = '30.' + str(pipLoc) + 'f'
            tp = format(tp, fstr).strip()
            sl = format(sl, fstr).strip()
            entry = format(entry, fstr).strip()
		    pipLoc = self.getPipSize(ins)
            pipVal = 10**(-pipLoc+1)
            expiry = datetime.datetime.now() + datetime.timedelta(hours = 6)
            args = {
                 'order': {
                     'instrument': ins,
                     'units': units,
                     'price': entry,
                     'type': 'STOP',
                     'timeInForce': 'GTD',
                     'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                     'takeProfitOnFill': {
                         'price': tp,
                         'timeInForce': 'GTC'},
                     'stopLossOnFill': {
                         'price': sl,
                         'timeInForce': 'GTC'
                         }}}
        if [lhll, low12, atrcond ] == [True, True, True]:
            condition_triggered.append('A1')
			entry = np.float(self.candles[1].get('h')) + pipVal + spread
            tp = entry + direction * pipVal * self.settings.get('takeProfit')
            sl = entry - direction * pipVal * self.settings.get('stopLoss')
            units = self.settings.get('units')
            # round off for the v20 api to accept the stops
            fstr = '30.' + str(pipLoc) + 'f'
            tp = format(tp, fstr).strip()
            sl = format(sl, fstr).strip()
            entry = format(entry, fstr).strip()
		    pipLoc = self.getPipSize(ins)
            pipVal = 10**(-pipLoc+1)
            expiry = datetime.datetime.now() + datetime.timedelta(hours = 3)
            args = {
                 'order': {
                     'instrument': ins,
                     'units': units,
                     'price': entry,
                     'type': 'STOP',
                     'timeInForce': 'GTD',
                     'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                     'takeProfitOnFill': {
                         'price': tp,
                         'timeInForce': 'GTC'},
                     'stopLossOnFill': {
                         'price': sl,
                         'timeInForce': 'GTC'
                         }}}
        
        ticket = self.oanda.order.create(
                  self.settings.get('account_id'), **args)
        ticket_json = json.loads(ticket.raw_body)
        #print(ticket_json)
