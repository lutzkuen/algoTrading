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
Backtesting engine for the algoTrader
Author: Lutz Kuenneke, 25.08.2018
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
import algoTrader.triangle as triangle
import algoTrader.divergence as divergence
import algoTrader.sentiment as sentiment
import matplotlib
matplotlib.use('Agg')
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import dataset

class controller(object):
 def __init__(self, confname,_type):
  config = configparser.ConfigParser()
  config.read(confname)
  self.output = open('/home/ubuntu/algoTrading/btresults.dat','w')
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
  #self.settings['imdir'] = config.get(_type,'imdir')
  self.settings['db_location'] = config.get(_type, 'db_location')
  self.settings['warmup']= int(config.get(_type, 'warmup'))
  self.db = dataset.connect(self.settings.get('db_location'))
  self.table = self.db['dailycandles']
  self.results = self.db['backtest_result']
  self.training_days = sorted([self.str2date(row['date']) for row in self.table.distinct('date')], key = lambda x: self.str2int(self.date2str(x)))
  if len(self.training_days) < self.settings.get('warmup'):
   print('Not enough training data')
   code.interact(banner='', local=locals())
  self.now = self.training_days[self.settings.get('warmup')]
  self.nowid = self.settings.get('warmup')
  self.lastday = self.training_days[-1]
  print('Starting Backtest on ' + self.date2str(self.now))
  self.trades = []
  self.orders = []
  self.cpers = {}
  self.indicators = [ divergence.indicator(self) , triangle.indicator(self) ] # sentiment is not backtest ready # sentiment is not backtest ready
 def runBacktest(self):
  while self.step():
   print('Backtest ' + self.date2str(self.now))
   time.sleep(60*60) # do one day per hour. This is meant to keep the CPU Credit usage on AWS down
 def evalOrder(self, ins):
  spread = self.getSpread(ins)
  if len(self.orders) == 0:
   return 0
  nexttime = self.date2str(self.training_days[self.nowid+1])
  nextbar = self.table.find_one(ins = ins, date = nexttime)
  if not nextbar:
   return None
  high = float(nextbar['high'])
  low = float(nextbar['low'])
  ope = float(nextbar['open'])
  close = float(nextbar['close'])
  # get the results now
  order = self.orders[0]
  units = int(order['order'].get('units'))
  entry = float(order['order'].get('price'))
  tp = float(order['order'].get('takeProfitOnFill').get('price'))
  sl = float(order['order'].get('stopLossOnFill').get('price'))
  typ = order['order'].get('type')
  pl = 0
  if typ == 'STOP':
   if units > 0: # buy stop
    if high > entry and low < sl: # assume a stop out
     pl += -units * (spread + entry - sl )
     return pl
    if high > tp:
     pl += units * ( tp - entry - spread )
     return pl
    if high > entry:
     pl += units * ( close - entry - spread )
     return pl
   else: # buy stop
    if low < entry and high > sl: # assume a stop out
     pl += -units * (spread + sl - entry )
     return pl
    if low < tp:
     pl += units * ( entry - tp - spread )
     return pl
    if low < entry:
     pl += units * ( entry - close - spread )
     return pl
  if typ == 'LIMIT':
   if units > 0: # buy stop
    if low < entry and low < sl: # assume a stop out
     pl += -units * (spread + entry - sl )
     return pl
    if low < entry and high > tp:
     pl += units * ( tp - entry - spread )
     return pl
    if low < entry:
     pl += units * ( close - entry - spread )
     return pl
   else: # buy stop
    if high > entry and high > sl: # assume a stop out
     pl += -units * (spread + sl - entry )
     return pl
    if low < tp and high > entry:
     pl += units * ( entry - tp - spread )
     return pl
    if high > entry:
     pl += units * ( entry - close - spread )
     return pl
  return 0
 def pl2db(self,ins,system,pl):
  dat = self.date2str(self.now)
  plstatement = {'date': dat, 'ins': ins, 'system': system, 'pl': pl}
  self.results.upsert(plstatement,['date','ins','system'])
 def step(self):
  for ins in self.allowed_ins:
   self.orders = []
   self.checkIns(ins.name)
  self.nowid += 1
  self.now = self.training_days[self.nowid]
  if self.nowid < (len(self.training_days)-1):
   return True
  return False
 def str2date(self, st):
  return datetime.datetime.strptime(st,'%Y-%m-%d')
 def date2str(self, dat):
  return dat.strftime('%Y-%m-%d')
 def str2int(self, st):
  ii = ''
  for s in st.split('-'):
   ii += s
  return int(ii)
 def drawImage(self, ins, candles, lines):
  return None # image drawing disabled in BT for now
  fig = plt.figure()
  ax1 = plt.subplot2grid((1,1), (0,0))
  ohlc = []
  n = 0
  for c in candles:
   #candle = converter(datetime.datetime.strptime(c.get('time')[:10],'%Y-%m-%d')), float(c.get('mid').get('o')), float(c.get('mid').get('h')), float(c.get('mid').get('l')), float(c.get('mid').get('c')), int(c.get('volume'))
   candle = n, float(c.get('mid').get('o')), float(c.get('ask').get('h')), float(c.get('bid').get('l')), float(c.get('mid').get('c')), int(c.get('volume'))
   ohlc.append(candle)
   n+=1
  candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#77d879', colordown='#db3f3f')
  for line in lines:
   plt.plot(line.get('xarr'),line.get('yarr'))
  for label in ax1.xaxis.get_ticklabels():
   label.set_rotation(45)
  #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  #ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
  ax1.grid(True)
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.title(ins)
  #plt.legend()
  plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
  imname = self.settings.get('imdir') + '/' + ins + '.pdf'
  fig.savefig(imname, bbox_inches='tight')
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
  price = self.getPrice(ins) # assume constant spread. We do not use pip multiple since this can cause overly optimistic estimations
  if price:
   return 0.0005*price
  else:
   return 999999999
 def getPrice(self, ins):
  row = self.table.find_one(ins=ins, date = self.date2str(self.now))
  try:
   cl = float(row['close'])
  except:
   cl = None
  return cl
 def getRSI(self, ins, granularity, numCandles):
  candles = self.getCandles(ins,granularity,numCandles)
  delta = [float(c.get('mid').get('c')) - float(c.get('mid').get('o')) for c in candles]
  sup = sum([upval for upval in delta if upval > 0])
  flo = sum([upval for upval in delta if upval < 0])
  if flo == 0:
   return 1
  rsi = 1-1/(1+sup/flo)
  return rsi
 def getCandles(self,ins,granularity,numCandles):
  if granularity != 'D': # only 'D' possible
   return None
  ckey = ins + '_' + granularity
  if ckey in self.cpers.keys():
   #code.interact(banner='', local=locals())
   if len(self.cpers[ckey].get('candles')) >= numCandles:
    candles = self.cpers[ckey].get('candles')[-numCandles:]
    return candles
  nowint = self.str2int(self.date2str(self.now))
  # the following line will: 1. read all lines for ins from db, 2. dismiss all which are in the future, 3. sort asc by date, 4. take the last numCandles
  candles = sorted([{'date': self.str2date(row['date']), 'mid': { 'o': float(row['open']), 'h': float(row['high']), 'l': float(row['low']), 'c': float(row['close'])}} for row in self.table.find(ins = ins) if self.str2int(row['date'])], key = lambda x: self.str2int(self.date2str(x['date'])))[-numCandles:]
  return candles

 def checkIns(self, ins):
  for indicator in self.indicators:
   indicator.checkIns(ins)
   pl = self.evalOrder(ins)
   if pl:
    self.pl2db(ins,indicator.name,pl)
 def createOrder(self, args):
  self.orders.append(args)
  return('Opening ' + str(args['order'].get('units')) + ' ' + str(args['order'].get('instrument')))
