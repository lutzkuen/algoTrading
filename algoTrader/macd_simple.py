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
  self.minbars = 10
  self.controller = controller
  self.name = 'macd_simple'
 def checkIns(self, ins):
  if len([trade for trade in self.controller.trades if trade.instrument == ins]) > 0:
   #print('Skipping ' + ins + ' found open trade')
   return None
  price = self.controller.getPrice(ins)
  if not price:
   return None
  spread = self.controller.getSpread(ins)
  pipLoc = self.controller.getPipSize(ins)
  pipVal = 10 ** (-pipLoc + 1)
  moveout = 2
  granularity = 'D'
  numCandles = 40
  macd = self.controller.getMACD(26,12,9,0,ins,'D')
  if not macd:
   #triangle = self.getTriangle(ins,'H4',180,spread)
   #if not triangle:
   return None # could not get triangle formation
  price = self.controller.getPrice(ins)
  if macd < 0:
   sl = self.controller.getMIN(ins,'D',3)
   tp = price + ( price - sl )/0.681
   units = self.controller.getUnits(abs(sl-price),ins)
  else:
   sl = self.controller.getMAX(ins,'D',3)
   tp = price - ( sl - price )/0.681
   units = -self.controller.getUnits(abs(sl-price),ins)

  fstr = '30.' + str(pipLoc) + 'f'
  tp = format(tp, fstr).strip()
  sl = format(sl, fstr).strip()
  expiry = datetime.datetime.now() + datetime.timedelta(days=1)
  args = {'order': {
      'instrument': ins,
      'units': units,
      'type': 'MARKET',
      'price': price,
      'takeProfitOnFill': {'price': tp, 'timeInForce': 'GTC'},
      'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'},
      }}
  ticket = self.controller.createOrder(args)
