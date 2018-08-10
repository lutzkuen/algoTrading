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
import math
import datetime
import numpy as np

def tdelta_to_float(tdelta):
 secsperday = 24*60*60
 return float(tdelta.days + tdelta.seconds/secsperday)

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
  self.settings['risk_factor'] = int(config.get('triangle', 'risk_factor'))
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

 def getTriangle(self,ins,granularity,numCandles):
  request = Request('GET',
                    '/v3/instruments/{instrument}/candles?count={count}&price={price}&granularity={granularity}'
                    )
  request.set_path_param('instrument', ins)
  request.set_path_param('count', numCandles)
  request.set_path_param('price', 'MBA')
  request.set_path_param('granularity', granularity)
  response = self.oanda.request(request)
  candles = json.loads(response.raw_body).get('candles')
  upperFractals = []
  lowerFractals = []
  fractRange = 2
  for i in range(fractRange,len(candles)-fractRange):
   isupper = True
   islower = True
   for k in range(i-fractRange,i+fractRange):
    if i == k:
     continue
    if float(candles[i].get('ask').get('h')) < float(candles[k].get('ask').get('h')):
     isupper = False
    if float(candles[i].get('bid').get('l')) > float(candles[k].get('bid').get('l')):
     islower = False
   if isupper:
    upperFractals.append(candles[i])
   if islower:
    lowerFractals.append(candles[i])
   if len(upperFractals) < 2 or len(lowerFractals) < 2:
    print('Not enough Fractal points in set')
    return None
  x = datetime.datetime.now() 
  # get the upper line 
  x1 = datetime.strptime(upperFractals[-2].get('time'),'%Y-%m-%dT%H:%M:%S.%fZ')
  y1 = float(upperFractals[-2]).get('ask').get('h')
  x2 = datetime.strptime(upperFractals[-1].get('time'),'%Y-%m-%dT%H:%M:%S.%fZ')
  y2 = float(upperFractals[-1]).get('ask').get('h')
  tdelta = x2 - x1
  mup = (y2-y1)/tdelta_to_float(tdelta)
  fupper = y1 + tdelta_to_float(x-x1)*mup
  # get the lower line
  x1 = datetime.strptime(lowerFractals[-2].get('time'),'%Y-%m-%dT%H:%M:%S.%fZ')
  y1 = float(lowerFractals[-2]).get('ask').get('h')
  x2 = datetime.strptime(lowerFractals[-1].get('time'),'%Y-%m-%dT%H:%M:%S.%fZ')
  y2 = float(lowerFractals[-1]).get('ask').get('h')
  tdelta = x2 - x1
  mlow = (y2-y1)/tdelta_to_float(tdelta)
  flower = y1 + tdelta_to_float(x-x1)*mlow
  if mup > 0 or mlow < 0:
   return None # in this case not a triangle formation  
  
  return [flower, fupper]

 def checkIns(self, ins):
  price = self.getPrice(ins)
  pipLoc = self.getPipSize(ins)
  pipVal = 10 ** (-pipLoc + 1)
  moveout = 4
  granularity = 'D1'
  numCandles = 50
  triangle = self.getTriangle(ins,granularity,numCandles)
  if not triangle:
   return None # could not get triangle formation
  
  upperentry = triangle[1]+moveout*pipVal
  lowerentry = triangle[0]-moveout*pipVal
  sl = (triangle[1]+triangle[0])/2
  tpupper = upperentry + (upperentry - sl)/0.618 # some serious FIB stuff
  tplower = lowerentry + (lowerentry - sl)/0.618 # some serious FIB stuff
  upperunits = self.getUnits(abs(sl-upperentry),ins)
  lowerunits = self.getUnits(abs(sl-lowerentry),ins)
  

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
      'trailingStopLossOnFill': {'distance': sldist, 'timeInForce': 'GTC'},
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
      'trailingStopLossOnFill': {'distance': sldist, 'timeInForce': 'GTC'},
      'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'},
      }}
  ticket = self.oanda.order.create(self.settings.get('account_id'
          ), **args)
  ticket_json = json.loads(ticket.raw_body)
  print(ticket_json)