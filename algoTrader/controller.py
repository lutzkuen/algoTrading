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
import algoTrader.triangle as triangle
import algoTrader.macd_simple as macd_simple
import algoTrader.triangleh4 as triangleh4
import algoTrader.triangleh4_lim as triangleh4_lim
import algoTrader.divergence as divergence
import algoTrader.sentiment as sentiment
import matplotlib
matplotlib.use('Agg')
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
  self.settings['minbars'] = int(config.get('triangle', 'minbars'))
  self.settings['maxbars'] = int(config.get('triangle', 'maxbars'))
  self.settings['moveout'] = int(config.get('triangle', 'moveout'))
  self.settings['tolerance'] = float(config.get('triangle', 'tolerance'))
  self.settings['granularity'] = config.get('triangle', 'granularity')
  self.settings['myfxbook_email'] = config.get('myfxbook','email')
  self.settings['myfxbook_pwd'] = config.get('myfxbook','pwd')
  self.settings['imdir'] = config.get(_type,'imdir')
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
  self.cpers = {}
  self.mtcounter = 0
  if _type in ['demo','backtest']:
   self.indicators = [ macd_simple.indicator(self) ]# divergence.indicator(self) , triangle.indicator(self)]#, sentiment.indicator(self) ]
   #self.indicators = [ sentiment.indicator(self)]
  if _type == 'live':
   self.indicators = [ triangleh4.indicator(self), triangleh4_lim.indicator(self)]#, sentiment.indicator(self) ]
  if _type == 'demoh':
   self.indicators = [ triangleh4.indicator(self), triangleh4_lim.indicator(self) ]
 def updateMTcounter(self):
  orders = \
      self.oanda.order.list(self.settings.get('account_id'
          ))
  orders = json.loads(orders.raw_body)
  for order in orders.get('orders'):
   try:
    self.mtcounter = max(int(order.get('tradeClientExtensions').get('id')),self.mtcounter)+1
   except:
    self.mtcounter += 1000
  
 def drawImage(self, ins, candles, lines):
  if self.settings.get('imdir')=='None':
   return None
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
  if not eurusd:
   return None
  for ins in self.allowed_ins:
      if leadingCurr in ins.name and 'USD' in ins.name:
          price = self.getPrice(ins.name)
          if not price:
              return None
          if ins.name.split('_')[0] == 'USD':
              return price / eurusd
          else:
              return 1.0 / (price * eurusd)
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
  if not conversion:
   trailingCurr = ins.split('_')[1]
   conversion = self.getConversion(trailingCurr)
   if conversion:
    conversion = conversion/price
  multiplier = min(price / dist, 100) # no single trade can be larger than the account NAV
  if not conversion:
      print('CRITICAL: Could not convert ' + leadingCurr + '_'+trailingCurr+ ' to EUR')
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
  fc = 0.681
  for trade in self.trades:
   price = self.getPrice(trade.instrument)
   macd_0 = self.getMACD(26, 12,9, 0, trade.instrument, 'D')
   macd_1 = self.getMACD(26, 12,9, 1, trade.instrument, 'D')
   print(trade.instrument + ' -- prev. MACD: ' + str(macd_1) + ' now MACD: ' + str(macd_0))
   if trade.currentUnits > 0:
    # check wether we should exit
    if macd_0 < macd_1:
     print('Closing ' + trade.instrument + ' on ' + str(macd_0) + ' < ' + str(macd_1))
     self.oanda.trade.close(self.settings.get('account_id'), trade.id)
     continue
    newSL = self.getMIN(trade.instrument,'D',3)
    newTP = price + ( price - newSL )/fc
    slo = trade.stopLossOrder
    if newSL > float(slo.price):
     print(trade.instrument + ' new SL ' + str(newSL))
     self.oanda.order.cancel(self.settings.get('account_id'), slo.id)
     pipLoc = self.getPipSize(trade.instrument)
     fstr = '30.' + str(pipLoc) + 'f'
     newSL = format(newSL, fstr).strip()
     newTP = format(newTP, fstr).strip()
     args = { 'stopLoss': {
                'instrument': trade.instrument,
                'units': -trade.currentUnits,
                'price': newSL,
                'type': 'STOP_LOSS'}}
     response = self.oanda.trade.set_dependent_orders(self.settings.get('account_id'), trade.id, **args)
     print(str(json.loads(response.raw_body)))
     args = { 'takeProfit': {
                'instrument': trade.instrument,
                'units': -trade.currentUnits,
                'price': newTP,
                'type': 'TAKE_PROFIT'}}
     response = self.oanda.trade.set_dependent_orders(self.settings.get('account_id'), trade.id, **args)
     print(str(json.loads(response.raw_body)))
   if trade.currentUnits < 0:
    if macd_0 > macd_1:
     print('Closing ' + trade.instrument + ' on ' + str(macd_0) + ' > ' + str(macd_1))
     self.oanda.trade.close(self.settings.get('account_id'), trade.id)
     continue
    newSL = self.getMAX(trade.instrument,'D',3)
    newTP = price - ( newSL - price )/fc
    slo = trade.stopLossOrder
    if newSL < float(slo.price):
     print(trade.instrument + ' new SL ' + str(newSL))
     self.oanda.order.cancel(self.settings.get('account_id'), slo.id)
     pipLoc = self.getPipSize(trade.instrument)
     fstr = '30.' + str(pipLoc) + 'f'
     newSL = format(newSL, fstr).strip()
     newTP = format(newTP, fstr).strip()
     args = { 'stopLoss': {
                'instrument': trade.instrument,
                'units': -trade.currentUnits,
                'price': newSL,
                'type': 'STOP_LOSS'}}
     response = self.oanda.trade.set_dependent_orders(self.settings.get('account_id'), trade.id, **args)
     print(str(json.loads(response.raw_body)))
     args = { 'takeProfit': {
                'instrument': trade.instrument,
                'units': -trade.currentUnits,
                'price': newTP,
                'type': 'TAKE_PROFIT'}}
     response = self.oanda.trade.set_dependent_orders(self.settings.get('account_id'), trade.id, **args)
     print(str(json.loads(response.raw_body)))
   #rsi = self.getRSI(trade.instrument, 'D', 5)
   #rsishort = self.getRSI(trade.instrument, 'D', 3)
   #if trade.currentUnits < 0 and ( rsi > 0.5 or rsishort < 0.2):
   # print('Closing ' + trade.instrument + '(' + str(trade.currentUnits) + ')')
   # self.oanda.trade.close(self.settings.get('account_id'), trade.id)
   #if trade.currentUnits > 0 and ( rsi < 0.5 or rsishort > 0.8 ):
   # print('Closing ' + trade.instrument + '(' + str(trade.currentUnits) + ')')
   # self.oanda.trade.close(self.settings.get('account_id'), trade.id)
 def getMIN(self, ins, granularity, numCandles):
  candles = self.getCandles(ins,granularity,numCandles)
  return min([float(c.get('mid').get('l')) for c in candles])
 def getMAX(self, ins, granularity, numCandles):
  candles = self.getCandles(ins,granularity,numCandles)
  return max([float(c.get('mid').get('h')) for c in candles])
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
  ckey = ins + '_' + granularity
  if ckey in self.cpers.keys():
   #code.interact(banner='', local=locals())
   if len(self.cpers[ckey].get('candles')) >= numCandles:
    candles = self.cpers[ckey].get('candles')[-numCandles:]
    return candles
   
  request = Request('GET',
                    '/v3/instruments/{instrument}/candles?count={count}&price={price}&granularity={granularity}'
                    )
  request.set_path_param('instrument', ins)
  request.set_path_param('count', numCandles)
  request.set_path_param('price', 'MBA')
  request.set_path_param('granularity', granularity)
  response = self.oanda.request(request)
  self.cpers[ckey] = json.loads(response.raw_body)
  candles = json.loads(response.raw_body).get('candles')
  return candles
 def getMeanPrice(self, candle):
  #return float(candle.get('mid').get('c'))
  return (float(candle.get('mid').get('l')) + float(candle.get('mid').get('h')) + 2*float(candle.get('mid').get('c')))/4
 def getSMA(self, ins, period, shift, granularity):
  nc = period + shift
  candles = self.getCandles(ins, granularity, nc)
  if shift > 0:
   candles[-shift:] = [] # just drop the end to get the shift
  sma = 0
  wsum = 0
  for i in range(0,len(candles)):
   sma += self.getMeanPrice(candles[i])
   wsum += 1
  return sma/wsum
 def getEMA(self, ins, period, shift, granularity):
  nc = period + shift
  candles = self.getCandles(ins, granularity, nc)
  if shift > 0:
   candles[-shift:] = [] # just drop the end to get the shift
  alpha = 1/(period+1)
  ema = 0
  wsum = 0
  for i in range(0,len(candles)):
   w = math.exp(alpha*(i-len(candles)))
   ema += w*self.getMeanPrice(candles[i])
   wsum += w
  return ema/wsum
  
 def getMACD(self, slowperiod, fastperiod, avperiod, shift, ins, granularity):
  mcvec = []
  for i in range(avperiod):
   mcvec.append(self.getEMA(ins, fastperiod,shift+i, granularity) - self.getEMA(ins, slowperiod,shift+i, granularity))
  alpha = 1/(1+avperiod)
  emamacd = 0
  wsum = 0
  for i in range(0,avperiod):
   w = math.exp(alpha*(i-avperiod))
   j = avperiod - i - 1
   emamacd += w*mcvec[j]
   wsum += w
  emamacd /= wsum
  return mcvec[0]-emamacd # this returns actually the histogram. Its all pretty weird but the final number is useful on a D1 Basis
  
 def checkIns(self, ins):
  for indicator in self.indicators:
   indicator.checkIns(ins)
 def sendOrder(self, args):
  ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
  return json.loads(ticket.raw_body)
 def createOrder(self, args):
  # add client Extensions for MT4
  ins = args['order'].get('instrument')
  macd_0 = self.getMACD(26, 12,9, 0, ins, 'D')
  macd_1 = self.getMACD(26, 12,9, 1, ins, 'D')
  allowed = False
  sma10 = self.getSMA(ins, 50, 0, 'D')# 10 week
  sma20 = self.getSMA(ins, 100, 0, 'D')# 20 week
  if int(args['order'].get('units')) > 0:
   if macd_0 > macd_1:#Momentum
    if sma10 > sma20:
     if float(args['order'].get('price')) > sma20:#trend
      if float(args['order'].get('price')) < sma10:#trend
       allowed = True
  if int(args['order'].get('units')) < 0:
   if macd_0 < macd_1:
    if sma10 < sma20:
     if float(args['order'].get('price')) < sma20:
      if float(args['order'].get('price')) > sma10:
       allowed = True
  #self.updateMTcounter()
  #args['order']['clientExtensions'] = { 'id': str(self.mtcounter), 'tag': '0' }
  #args['order']['tradeClientExtensions'] = { 'id': str(self.mtcounter), 'tag': '0' }
  if allowed:
   ticket = self.sendOrder(args)
   return ticket
