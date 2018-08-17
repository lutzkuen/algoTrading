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
Places orders to trade the triangle
Use at own risk
Author: Lutz Kuenneke, 12.07.2018
"""
import json
import time
import v20
from v20.request import Request
import requests
import configparser
import code
import math
import datetime
import numpy as np

def tdelta_to_float(tdelta):
 secsperday = 24*60*60
 return float(tdelta.days)

class controller(object):
 def __init__(self, confname,_type):
  config = configparser.ConfigParser()
  config.read(confname)
  self.settings = {}
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
  self.trades = \
      self.oanda.trade.list_open(self.settings.get('account_id'
          )).get('trades', '200')
  self.minbars = 10
  self.cpers = {}
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
 def getSpread(self, ins):
  args = {'instruments': ins}
  nprice = self.oanda.pricing.get(self.settings.get('account_id'
          ), **args)
  pobj = json.loads(nprice.raw_body)
  spread = abs(float(pobj.get('prices')[0].get('bids')[0].get('price'
           )) - float(pobj.get('prices')[0].get('asks'
           )[0].get('price')))
  return spread

 def getPrice(self, ins):
  args = {'instruments': ins}
  nprice = self.oanda.pricing.get(self.settings.get('account_id'
          ), **args)
  pobj = json.loads(nprice.raw_body)
  price = (float(pobj.get('prices')[0].get('bids')[0].get('price'
           )) + float(pobj.get('prices')[0].get('asks'
           )[0].get('price'))) / 2.0
  return price
 def manageTrades(self):
  # this method checks all open trades, moves stops and closes were there was no favourable movement
  for trade in self.trades:
   rsi = self.getRSI(trade.instrument, 'D', 5)
   rsishort = self.getRSI(trade.instrument, 'D', 3)
   if trade.currentUnits < 0 and ( rsi > 0.5 or rsishort < 0.2):
    print('Closing ' + trade.instrument + '(' + str(trade.currentUnits) + ')')
    self.oanda.trade.close(self.settings.get('account_id'), trade.id)
   if trade.currentUnits > 0 and ( rsi < 0.5 or rsishort > 0.8 ):
    print('Closing ' + trade.instrument + '(' + str(trade.currentUnits) + ')')
    self.oanda.trade.close(self.settings.get('account_id'), trade.id)
 def getRSI(self, ins, granularity, numCandles):
  request = Request('GET',
                    '/v3/instruments/{instrument}/candles?count={count}&price={price}&granularity={granularity}'
                    )
  request.set_path_param('instrument', ins)
  request.set_path_param('count', numCandles)
  request.set_path_param('price', 'M')
  request.set_path_param('granularity', granularity)
  #print(granularity)
  response = self.oanda.request(request)
  #print(response.raw_body)
  try:
   candles = json.loads(response.raw_body).get('candles')[-numCandles:]
  except:
   print('Failed to get RSI')
   return None
  delta = [float(c.get('mid').get('c')) - float(c.get('mid').get('o')) for c in candles]
  sup = sum([upval for upval in delta if upval > 0])
  flo = sum([upval for upval in delta if upval < 0])
  if flo == 0:
   return 1
  rsi = 1-1/(1+sup/flo)
  return rsi
 def getTriangle(self,ins,granularity,numCandles,spread):
  if numCandles < self.minbars:
   return None
  if ins in self.cpers.keys():
   candles = self.cpers[ins].get('candles')[-numCandles:]
   #candles = [candle for candle in self.cpers[ins].get('candles')[-numCandles:] if bool(candle.get('complete') )]# Der aktuell begonnene Tag soll ignoriert werden
  else:
   
   request = Request('GET',
                     '/v3/instruments/{instrument}/candles?count={count}&price={price}&granularity={granularity}'
                     )
   request.set_path_param('instrument', ins)
   request.set_path_param('count', numCandles)
   request.set_path_param('price', 'MBA')
   request.set_path_param('granularity', granularity)
   #print(granularity)
   response = self.oanda.request(request)
   #print(response.raw_body)
   self.cpers[ins] = json.loads(response.raw_body)
   candles = json.loads(response.raw_body).get('candles')
   #candles = [candle for candle in json.loads(response.raw_body).get('candles') if bool(candle.get('complete') )]# Der aktuell begonnene Tag soll ignoriert werden
  if not candles:
   return None
  upperFractals = []
  lowerFractals = []
  fractRange = 2
  highs = [candle.get('ask').get('h') for candle in candles[:-1]]
  lows = [candle.get('bid').get('l') for candle in candles[:-1]]
  x = len(candles) #datetime.datetime.strptime(candles[-1].get('time').split('.')[0],'%Y-%m-%dT%H:%M:%S')
  # get the upper line 
  y1 = 0.0
  mbest = -math.inf
  fupper = None
  n = 0
  for candle in candles:
   if float(candle.get('ask').get('h')) >= y1:
    x1 = n # datetime.datetime.strptime(candle.get('time').split('.')[0],'%Y-%m-%dT%H:%M:%S')
    y1 = float(candle.get('ask').get('h'))
    mbest = -math.inf
    fupper = None
    confirmed = False
   else:
    x2 = n # datetime.datetime.strptime(candle.get('time').split('.')[0],'%Y-%m-%dT%H:%M:%S')
    y2 = float(candle.get('ask').get('h'))
    tdelta = x2 - x1
    mup = (y2-y1)/float(tdelta)
    if mup > mbest:
     mbest = mup
     fupper = y1 + float(x-x1)*mup
     confirmed = False
     nc = 0
     for cc in candles:
      if nc == x1 or nc == x2:
       continue
      festim = y1 + float(nc-x1)*mup
      if festim - float(cc.get('ask').get('h')) < 5*spread:
       confirmed = True
      nc += 1
   n += 1
  # get the lower line
  y1 = math.inf
  mbest = math.inf
  flower = None
  n = 0
  for candle in candles:
   if float(candle.get('bid').get('l')) <= y1:
    x1 = n# datetime.datetime.strptime(candle.get('time').split('.')[0],'%Y-%m-%dT%H:%M:%S')
    y1 = float(candle.get('bid').get('l'))
    mbest = math.inf
    flower = None
    confirmedl = False
   else:
    x2 = n#  datetime.datetime.strptime(candle.get('time').split('.')[0],'%Y-%m-%dT%H:%M:%S')
    y2 = float(candle.get('bid').get('l'))
    tdelta = x2 - x1
    mup = (y2-y1)/float(tdelta)
    if mup < mbest:
     mbest = mup
     flower = y1 + float(x-x1)*mup
     confirmedl = False
     nc = 0
     for cc in candles:
      if nc == x1 or nc == x2:
       continue
      festim = y1 + float(nc-x1)*mup
      if float(cc.get('bid').get('l')) - festim < 5*spread:
       confirmedl = True
      nc += 1
   n += 1
  #print(ins + ' ' + str(flower) + ' ' + str(fupper))
  if not fupper or not flower or not confirmed or not confirmedl:
   return self.getTriangle(ins,granularity,numCandles-1,spread)
  
  return [flower, fupper]

 def checkIns(self, ins):
  if len([trade for trade in self.trades if trade.instrument == ins]) > 0:
   print('Skipping ' + ins + ' found open trade')
   return None
  price = self.getPrice(ins)
  spread = self.getSpread(ins)
  pipLoc = self.getPipSize(ins)
  pipVal = 10 ** (-pipLoc + 1)
  moveout = 2
  granularity = 'D'
  numCandles = 40
  triangle = self.getTriangle(ins,granularity,numCandles,spread)
  if not triangle:
   #triangle = self.getTriangle(ins,'H4',180,spread)
   #if not triangle:
   return None # could not get triangle formation
  
  upperentry = triangle[1]+moveout*spread
  lowerentry = triangle[0]-moveout*spread
  sl = (triangle[1]+triangle[0])/2
  tpupper = upperentry + (upperentry - sl)/0.618 # some serious FIB stuff
  tplower = lowerentry + (lowerentry - sl)/0.618 # some serious FIB stuff
  upperunits = self.getUnits(abs(sl-upperentry),ins)
  lowerunits = -self.getUnits(abs(sl-lowerentry),ins)
  if price > upperentry or price < lowerentry or upperunits == 0 or lowerunits == 0:
   print('Skipping ' + ins + '. ' + str(price) + ' ' + str(upperentry) + ' ' + str(lowerentry) + ' ' + str(upperunits) + ' ' + str(lowerunits))
   return None # skip if not inside triangle

  fstr = '30.' + str(pipLoc) + 'f'
  tpupper = format(tpupper, fstr).strip()
  tplower = format(tplower, fstr).strip()
  sl = format(sl, fstr).strip()
  #sldist = format(sldist, fstr).strip()
  upperentry = format(upperentry, fstr).strip()
  lowerentry = format(lowerentry, fstr).strip()
  expiry = datetime.datetime.now() + datetime.timedelta(days=1)
  args = {'order': {
      'instrument': ins,
      'units': upperunits,
      'price': upperentry,
      'type': 'STOP',
      'timeInForce': 'GTD',
      'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
      'takeProfitOnFill': {'price': tpupper, 'timeInForce': 'GTC'},
      'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'},
      }}
  ticket = self.oanda.order.create(self.settings.get('account_id'
          ), **args)
  ticket_json = json.loads(ticket.raw_body)
  print(ticket_json)
  args = {'order': {
      'instrument': ins,
      'units': lowerunits,
      'price': lowerentry,
      'type': 'STOP',
      'timeInForce': 'GTD',
      'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
      'takeProfitOnFill': {'price': tplower, 'timeInForce': 'GTC'},
      'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'},
      }}
  ticket = self.oanda.order.create(self.settings.get('account_id'
          ), **args)
  ticket_json = json.loads(ticket.raw_body)
  print(ticket_json)
