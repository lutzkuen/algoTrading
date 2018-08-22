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


class indicator(object):
 def __init__(self, controller):
  self.minbars = 5
  self.controller = controller
 def getRSI(candles):
  delta = [float(c.get('mid').get('c')) - float(c.get('mid').get('o')) for c in candles]
  sup = sum([upval for upval in delta if upval > 0])
  flo = sum([upval for upval in delta if upval < 0])
  if flo == 0:
   return 1
  rsi = 1-1/(1+sup/flo)
 def getStoch(values):
  if min(values) < max(values):
   return (values[-1]-min(values))/(max(values)-min(values))
  else:
   return 0.5
 def getDivergence(self,ins,granularity,numCandles,window):
  if window <= numCandles:
   return None
  candles = self.controller.getCandles(ins, granularity, numCandles)
  if not candles:
   return None
  # get the RSI array
  rsi = [self.getRSI(candles[(i-window):window]) for i in range(window,numCandles)]
  stoch = [self.getStoch(rsi[(i-window):window]) for i in range(window,numCandles)]
  # periodic boundary conditions
  candles.append(candles[-2])
  candles.append(candles[-4])
  stoch.append(stoch[-2])
  stoch.append(stoch[-4])
  lows = []
  highs = []
  phigh = [c.get('ask').get('h') for c in candles]
  plow = [c.get('bid').get('l') for c in candles]
  for i in range(3,len(candles)-3):
   stochhigh = False
   stochlow = False
   pricehigh = False
   pricelow = False
   stochbefore = stoch[-(i+2):-i]
   phbef = phigh[-(i+2):-i]
   plbef = plow[-(i+2):-i]
   if i > 3:
    stochafter = stoch[-(i-1):-(i-3)]
    phaft = phigh[-(i-1):-(i-3)]
    plaft = plow[-(i-1):-(i-3)]
   else:
    stochafter = stoch[-(i-1):]
    phaft = phigh[-(i-1):]
    plaft = plow[-(i-1):]
   if stoch[-i] >= max(stochbefore) and stoch[-i] > max(stochafter):
    stochhigh = True
   if stoch[-i] <= min(stochbefore) and stoch[-i] < min(stochafter):
    stochlow = True
   if phigh[-i] >= max(phbef) and phigh[-i] > max(phaft):
    pricehigh = True
   if plow[-i] <= min(plbef) and plow[-i] < min(plhaft):
    pricelow = True
   if pricehigh and stochhigh:
    highs.append({'type': 'HH', 'index': i, 'l': candles[-i].get('mid').get('l'), 'h': candles[-i].get('mid').get('h'), 's': stoch[-i]})
   if pricelow and stochlow:
    lows.append({'type': 'LL', 'index': i, 'l': candles[-i].get('mid').get('l'), 'h': candles[-i].get('mid').get('h'), 's': stoch[-i]})
   if len(lows) >= 2 and len(highs) >= 2:
    break
  sma10 = np.mean([c.get('mid').get('c') for c in candles[-10:]])
  sma20 = np.mean([c.get('mid').get('c') for c in candles[-20:]])
  if len(lows) < 2 or len(highs) < 2:
   return None
  # check for hidden bearish after regular bearish
  if lows[1].get('s') > lows[0].get('s') and lows[1].get('l') > lows[0].get('l') and lows[1].get('h') > lows[0].get('l'):#regular bearish
   if highs[1].get('s') < highs[0].get('s') and highs[1].get('l') > highs[0].get('l') and highs[1].get('h') > highs[0].get('h'):
    # go short
    if sma20 > sma10:
     return [sma20, sma10]
  # check for hidden bullish after regular bullish
  if highs[1].get('s') < highs[0].get('s') and highs[1].get('l') < highs[0].get('l') and highs[1].get('h') < highs[0].get('h'):
   if lows[1].get('s') > lows[0].get('s') and lows[1].get('l') < lows[0].get('l') and lows[1].get('h') < lows[0].get('l'):
    if sma20 < sma10:
     return [sma20, sma10]
  return None
  
 def checkIns(self, ins):
  if len([trade for trade in self.controller.trades if trade.instrument == ins]) > 0:
   print('Skipping ' + ins + ' found open trade')
   return None
  price = self.controller.getPrice(ins)
  spread = self.controller.getSpread(ins)
  pipLoc = self.controller.getPipSize(ins)
  pipVal = 10 ** (-pipLoc + 1)
  moveout = 2
  granularity = 'D'
  numCandles = 100
  window = 14
  diverge = self.getDivergence(ins,granularity,numCandles,window)
  if not diverge:
   #triangle = self.getTriangle(ins,'H4',180,spread)
   #if not triangle:
   return None # could not get triangle formation
  
  sl = diverge[0]
  entry = diverge[1]
  tp = entry + ( entry - sl)/0.618
  units = self.controller.getUnits(abs(sl-entry),ins)
  if entry < sl:
   units = -units
   if price < entry:
    typ = 'LIMIT'
   else:
    typ = 'STOP'
  else:
   if price > entry:
    typ = 'LIMIT'
   else:
    typ = 'STOP'

  fstr = '30.' + str(pipLoc) + 'f'
  tp = format(tp, fstr).strip()
  sl = format(sl, fstr).strip()
  #sldist = format(sldist, fstr).strip()
  entry = format(entry, fstr).strip()
  expiry = datetime.datetime.now() + datetime.timedelta(days=1)
  args = {'order': {
      'instrument': ins,
      'units': units,
      'price': entry,
      'type': typ,
      'timeInForce': 'GTD',
      'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
      'takeProfitOnFill': {'price': tp, 'timeInForce': 'GTC'},
      'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'},
      }}
  ticket = self.controller.oanda.order.create(self.controller.settings.get('account_id'
          ), **args)
  ticket_json = json.loads(ticket.raw_body)
