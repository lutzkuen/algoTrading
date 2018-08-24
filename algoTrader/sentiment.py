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
  self.threshold = 0.2
  self.controller = controller
  self.myfxbook = None
  self.oandaOrderbook = None
 def getOanda(self, ins):
  request = Request('GET', '/v3/instruments/{instrument}/positionBook')
  request.set_path_param('instrument', ins)
  response = self.controller.oanda.request(request)
  try:
   resp = json.loads(response.raw_body)
  except:
   print('Failed to get oanda position book ' + ins)
   return None
  print(resp)
  print(resp.keys())
  if 'errorMessage' in resp.keys():
   return None
  netlong = np.sum([float(bucket.get('longCountPercent')) for bucket in resp.get('positionBook').get('buckets')])
  netshort = np.sum([float(bucket.get('shortCountPercent')) for bucket in resp.get('positionBook').get('buckets')])
  return netlong/(netlong + netshort)
 def getmyfx(self,ins):
  if not self.myfxbook:
   email = self.controller.settings.get('myfxbook_email')
   passwd = self.controller.settings.get('myfxbook_pwd')
   #login
   login_url = 'https://www.myfxbook.com/api/login.json?email=' + email + '&password=' + passwd
   response = requests.get(url = login_url)
   session = response.json().get('session')
   #print('Opened session ' + str(session))
   #get the community position book
   com_url = 'http://www.myfxbook.com/api/get-community-outlook.json?session=' + session
   response = requests.get(url = com_url)
   #code.interact(banner='', local=locals())
   try:
    position_book = response.json()
   except:
    print('Failed to get community outlook')
    return None
   self.myfxbook = position_book.get('symbols')
   logout_url = 'https://www.myfxbook.com/api/logout.json?session=' + session
   response = requests.get(url = logout_url)
  for sym in self.myfxbook:
   if sym.get('name') == 'AUDJPY':
    longpos = float(sym.get('longPositions'))
    shortpos = float(sym.get('shortPositions'))
    return longpos / ( longpos + shortpos )
 def getSentiment(self,ins):
  myfx = self.getmyfx(ins.replace('_',''))
  if not myfx:
   return None
  oanda = self.getOanda(ins)
  if not oanda:
   return None
  print(ins + ' ' + str(myfx) + ' ' + str(oanda))
  if myfx < self.threshold and oanda < self.threshold:
   return myfx * oanda
  if myfx > 1-self.threshold and oanda > 1-self.threshold:
   return 1-myfx*oanda
 def checkIns(self, ins):
  sent = self.getSentiment(ins)
  if not sent:
   return None
  granularity = 'D'
  numCandles = 20
  candles = self.controller.getCandles(ins, granularity, numCandles)
  sma10 = np.mean([float(c.get('mid').get('c')) for c in candles[-10:]])
  sma20 = np.mean([float(c.get('mid').get('c')) for c in candles[-20:]])
  price = self.controller.getPrice(ins)
  pipLoc = self.controller.getPipSize(ins)
  lower = min([sma10, sma20])
  higher = max([sma10, sma20])
  if sent < self.threshold:
   if price < lower:
    entry = lower
    typ = 'STOP'
    tp = higher
    sl = entry - 0.618*(tp-entry)
   else:
    if price < higher:
     entry = higher
     sl = lower
     tp = entry + ( entry - sl ) /0.618
     typ = 'STOP'
    else:
     entry = higher
     sl = lower
     tp = entry + ( entry - sl ) /0.618
     typ = 'LIMIT'
   units = self.controller.getUnits(abs(sl-entry),ins)
   fstr = '30.' + str(pipLoc) + 'f'
   sl = format(sl, fstr).strip()
   tp = format(tp, fstr).strip()
   entry = format(entry, fstr).strip()
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
   print(ticket_json)
  if sent > 1-self.threshold:
   if price > higher:
    entry = higher
    typ = 'STOP'
    tp = lower
    sl = entry + 0.618*(entry-tp)
   else:
    if price > lower:
     entry = lower
     sl = higher
     tp = entry - ( sl - entry  ) /0.618
     typ = 'STOP'
    else:
     entry = lower
     sl = higher
     tp = entry + ( sl - entry  ) /0.618
     typ = 'LIMIT'
   units = -self.controller.getUnits(abs(sl-entry),ins)
   fstr = '30.' + str(pipLoc) + 'f'
   sl = format(sl, fstr).strip()
   tp = format(tp, fstr).strip()
   entry = format(entry, fstr).strip()
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
   print(ticket_json)
