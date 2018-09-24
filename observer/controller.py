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
#import code
import math
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import dataset
import json


class controller(object):

    def __init__(self, confname, _type):
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
        self.db = dataset.connect('sqlite:////home/ubuntu/algoTrading/data/barsave.db')
        self.table = self.db['dailycandles']
    def retrieveData(self, numCandles):
        for ins in self.allowed_ins:
            candles = self.getCandles(ins.name,'D',numCandles)
            self.candlesToDB(candles, ins.name)
    def getPipSize(self, ins):
     pipLoc = [_ins.pipLocation for _ins in self.allowed_ins if _ins.name == ins]
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
    def data2sheet(self, write_raw = False, write_predict = True):
        inst = []
        statement = 'select distinct ins from dailycandles order by ins;'
        for row in self.db.query(statement):
            inst.append(row['ins'])
        dates =[]
        statement = 'select distinct date from dailycandles order by date;'
        for row in self.db.query(statement):
            #if row['date'][:4] == year:
            dates.append(row['date'])
        dstr =[]
        for date in dates:
            drow ={'date': date}
            for ins in inst:
                icandle = self.table.find_one(date = date, ins = ins)
                if not icandle:
                    print('Candle does not exist ' + ins +' '+ str(date))
                    drow[ins+'_vol'] = -1
                    drow[ins+'_open'] = -1
                    drow[ins+'_close'] = -1
                    drow[ins+'_high'] = -1
                    drow[ins+'_low'] = -1
                else:
                    drow[ins+'_vol'] = int(icandle['volume'])
                    drow[ins+'_open'] = float(icandle['open'])
                    drow[ins+'_close'] = float(icandle['close'])
                    drow[ins+'_high'] = float(icandle['high'])
                    drow[ins+'_low'] = float(icandle['low'])
            dstr.append(drow)
        df = pd.DataFrame(dstr)
        #code.interact(banner='', local=locals())
        if write_raw:
         print('Constructe DF with shape ' + str(df.shape))
         outname = '/home/ubuntu/data/cexport.csv'
         df.to_csv(outname)
        if write_predict:
         df.drop(['date'],1,inplace = True)
         volp = {}
         for col in df.columns:
          pvol, vprev = self.predictColumn(col, df)
          parts = col.split('_')
          #print(col)
          instrument = parts[0] + '_' + parts[1]
          typ = parts[2]
          if instrument in volp.keys():
           volp[instrument][typ] = pvol # store diff to prev day
          else:
           volp[instrument] = { typ: pvol }
          print(col + ' ' + str(pvol))
         
         #psort = sorted(volp, key = lambda x: x.get('relative'), reverse = True)
         outfile = open('/home/ubuntu/data/prices.csv','w')
         outfile.write('INSTRUMENT,HIGH,LOW,OPEN,CLOSE,VOLUME\n')
         for instr in volp.keys():
          outfile.write(str(instr) + ',' + str(volp[instr].get('high')) + ',' + str(volp[instr].get('low')) + ',' + str(volp[instr].get('open')) + ',' + str(volp[instr].get('close')) + ',' + str(volp[instr].get('vol')) + '\n')
         outfile.close()
    def predictColumn(self, pcol, df):
     x = np.array(df.values[:])
     y = np.array(df[pcol].values[:]) # make a deep copy to prevent data loss in future iterations
     vprev = y[-1]
     y = y[1:] # drop first line
     xlast = x[-1,:]
     x = x[:-1,:]# drop the last line
     regr = RandomForestRegressor()
     #remove missing lines from the training data
     i = 0
     while i < y.shape[0]:
      if y[i] < 0: # missing values are marked with -1
       y = np.delete(y,i)
       x = np.delete(x, (i), axis = 0)
      else:
       i += 1
     regr.fit(x,y)
     yp = regr.predict(xlast.reshape(1, -1))
     return yp[0], vprev
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
    def openLimit(self, ins):
     df = pd.read_csv('/home/ubuntu/data/prices.csv')
     op = df[df['INSTRUMENT'] == ins]['OPEN'].values[0]
     cl = df[df['INSTRUMENT'] == ins]['CLOSE'].values[0]
     hi = df[df['INSTRUMENT'] == ins]['HIGH'].values[0]
     lo = df[df['INSTRUMENT'] == ins]['LOW'].values[0]
     if hi < max([op, cl, hi, lo]) or lo > min([op, cl, hi, lo]): # inconsistent
      return None
     if cl > op:
      sl = lo - ( op - lo )
      tp = (hi + cl)/2
     else:
      sl = hi + ( hi - op )
      tp = (lo + cl)/2
     rr = abs((tp-op)/(sl-op))
     if rr < 1.5:# Risk-reward to low
      return None
     # if you made it here its fine, lets open a limit order
     units = self.getUnits(abs(sl-op),ins)
     if tp < sl:
      units *= -1
     pipLoc = self.getPipSize(ins)
     spread = self.getSpread(ins)
     if abs(sl-op) < 5*spread: # sl too small
      return None
     fstr = '30.' + str(pipLoc) + 'f'
     tp = format(tp, fstr).strip()
     sl = format(sl, fstr).strip()
     op = format(op, fstr).strip()
     expiry = datetime.datetime.now() + datetime.timedelta(days=1)
     args = {'order': {
     'instrument': ins,
     'units': units,
     'price': op,
     'type': 'LIMIT',
     'timeInForce': 'GTD',
     'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
     'takeProfitOnFill': {'price': tp, 'timeInForce': 'GTC'},
     'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'},
     }}
     ticket = self.oanda.order.create(self.settings.get('account_id'), **args)
     print(json.loads(ticket.raw_body))
    def getBTreport(self):
        outname = '/home/ubuntu/algoTrading/backtest.csv'
        outfile = open(outname,'w')
        outfile.write('SYSTEM,INS,PL\n')
        statement = 'select system, ins, sum(pl) from backtest_result where abs(pl) > 0  group by system, ins ;'
        for row in self.db.query(statement):
            oline = row['system'] + ',' + row['ins'] + ',' + str(row['sum(pl)'])
            outfile.write(oline+'\n')
        outfile.close()
