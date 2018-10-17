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
Candle logger and ML controller
Use at own risk
Author: Lutz Kuenneke, 26.07.2018
"""
import json
import time
try:
    import v20
    from v20.request import Request
    v20present = True
except ImportError:
    print('WARNING: V20 library not present. Connection to broker not possible')
    v20present = False
import requests
import code
    
import configparser
#import code
import math
#from sklearn.externals import joblib
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
#from tpot import TPOTRegressor
import dataset
import pickle
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.pipeline import make_pipeline


def getRangeInt(val,change,lower = -math.inf,upper = math.inf):
    lval = math.floor(val*(1-change))
    uval = math.ceil(val*(1+change))
    rang = []
    if lval < val and lval >= lower:
        rang.append(lval)
    rang.append(val)
    if uval > val and uval <= upper:
        rang.append(uval)
    return rang

def getRangeFlo(val,change,lower = -math.inf,upper = math.inf):
    lval = val*(1-change)
    uval = val*(1+change)
    rang = []
    if lval < val and lval >= lower:
        rang.append(lval)
    rang.append(val)
    if uval > val and uval <= upper:
        rang.append(uval)
    return rang    


def getGBimportances(gb, x, y):
    gb.fit(x,y)
    return gb.feature_importances_

class estim_pipeline(object):
    def __init__(self, percentile = 50, learning_rate = 0.1, n_estimators = 100, min_samples_split = 2, path = None, classifier = False):
        self.classifier = classifier
        if path: # read from disk 
            gb_path = path + '.gb'
            self.gb = pickle.load(open(gb_path,'rb'))
            perc_path = path + '.pipe'
            perc_attr = pickle.load(open(perc_path,'rb'))
            param_path = path + '.param'
            self.params = pickle.load(open(param_path,'rb'))
            score_func = lambda x, y: getGBimportances(self.gb, x, y)
            self.percentile = SelectPercentile(score_func=score_func, percentile=self.params.get('percentile'))
            self.percentile.scores_ = perc_attr.get('scores')
            self.percentile.pvalues_ = perc_attr.get('pvalues')
            #code.interact(banner='', local=locals())
        else:
            self.params = { 'percentile': percentile, 'learning_rate': learning_rate, 'n_estimators': n_estimators, 'min_samples_split': min_samples_split}
            if classifier:
                self.gb = GradientBoostingClassifier(learning_rate=learning_rate,  min_samples_split=min_samples_split, n_estimators=n_estimators)
            else:
                self.gb = GradientBoostingRegressor(learning_rate=learning_rate,  min_samples_split=min_samples_split, n_estimators=n_estimators)
            score_func = lambda x, y: getGBimportances(self.gb, x, y)
            self.percentile = SelectPercentile(score_func=score_func, percentile=percentile)
        self.pipeline = make_pipeline(
        self.percentile,
        self.gb
        )
        #print(self.pipeline.named_steps.keys())
    def get_params(self,deep = True):
        return self.params
    def fit(self, x,y,sample_weight = None):
        if self.classifier:
            self.pipeline.fit(x,y, gradientboostingclassifier__sample_weight = sample_weight )
        else:
            self.pipeline.fit(x,y, gradientboostingregressor__sample_weight = sample_weight )
        self.feature_importances_ = self.gb.feature_importances_
        return self
    def writeToDisk(self, path):
        gb_path = path + '.gb'
        pickle.dump(self.gb, open(gb_path,'wb'))
        pipe_path = path + '.pipe'
        #code.interact(banner='', local=locals())
        pipe_attr = { 'scores': self.percentile.scores_, 'pvalues': self.percentile.pvalues_ }
        pickle.dump(pipe_attr, open(pipe_path,'wb'))    
        param_path = path + '.param'
        pickle.dump(self.params, open(param_path,'wb'))
    def predict(self, x):
        return self.pipeline.predict(x)
    def score(self, x, y=None, sample_weight = None):
        return self.pipeline.score(x,y=y,sample_weight = sample_weight)
    def set_params(self, percentile = None, learning_rate = None, n_estimators = None,  min_samples_split = None):
        if percentile:
            self.percentile.set_params(percentile = percentile)
            self.params['percentile'] = percentile
        if learning_rate:
            self.gb.set_params(learning_rate = learning_rate)
            self.params['learning_rate'] = learning_rate
        if n_estimators:
            self.gb.set_params(n_estimators = n_estimators)
            self.params['n_estimators'] = n_estimators
        if min_samples_split:
            self.gb.set_params(min_samples_split = min_samples_split)
            self.params['min_samples_split'] = min_samples_split
        return self

class controller(object):
    def __init__(self, confname, _type):
        config = configparser.ConfigParser()
        config.read(confname)
        self.settings = {}
        self.settings['estim_path'] = config.get('data', 'estim_path')
        self.settings['prices_path'] = config.get('data','prices_path')
        if _type and v20present:
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
            self.trades = self.oanda.trade.list_open(self.settings.get('account_id')).get('trades', '200')
        self.db = dataset.connect(config.get('data','candle_path'))
        self.table = self.db['dailycandles']
        self.estimtable = self.db['estimators']
        self.importances = self.db['feature_importances']
    def retrieveData(self, numCandles, completed = True, upsert = False):
        for ins in self.allowed_ins:
            candles = self.getCandles(ins.name,'D',numCandles)
            self.candlesToDB(candles, ins.name, completed = completed, upsert = upsert)
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
    def candlesToDB(self, candles, ins, completed = True, upsert = False):
        for candle in candles:
         if (not bool(candle.get('complete')) ) and completed:
          continue
         time = candle.get('time')[:10]# take the YYYY-MM-DD part
         icandle = self.table.find_one(date = time, ins = ins)
         cobj = { 'ins': ins, 'date': time, 'open': candle.get('mid').get('o'), 'close': candle.get('mid').get('c'), 'high': candle.get('mid').get('h'), 'low': candle.get('mid').get('l'), 'volume': candle.get('volume'), 'complete': bool(candle.get('complete')) }
         if icandle:
          print(ins + ' ' + time + ' already in dataset')
          if upsert:
           self.table.upsert(cobj,['ins', 'date'])
          continue
         print('Inserting ' + str(cobj))
         self.table.insert(cobj)
    def getCandles(
        self,
        ins,
        granularity,
        numCandles):
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
    def data2sheet(self, write_raw = False, write_predict = True, improve_model = False, maxdate = None, newEstim = False, complete = True):
        inst = []
        if complete:
         c_cond = ' and complete = 1'
        else:
         c_cond = ' and complete in (0,1)'
        statement = 'select distinct ins from dailycandles order by ins;'
        for row in self.db.query(statement):
            inst.append(row['ins'])
        dates =[]
        if maxdate:
         statement = 'select distinct date from dailycandles where date <= ' + maxdate + c_cond +  ' order by date;'
        else:
         statement = 'select distinct date from dailycandles order by date ' + c_cond + ';'
        for row in self.db.query(statement):
            #if row['date'][:4] == year:
            dates.append(row['date'])
        dstr =[]
        if (not improve_model) and (not newEstim): # if we want to read only it is enough to take the last days
         dates = dates[-4:]
        #dates = dates[-30:] # use this line to decrease computation time for development
        for date in dates:
            # check whether the candle is from a weekday
            dspl = date.split('-')
            weekday = int(datetime.datetime(int(dspl[0]), int(dspl[1]), int(dspl[2])).weekday())
            if weekday == 4 or weekday == 5: # saturday starts on friday and sunday on saturday
                continue
            drow ={'date': date, 'weekday': weekday }
            for ins in inst:
                if complete:
                 icandle = self.table.find_one(date = date, ins = ins, complete = 1)
                else:
                 icandle = self.table.find_one(date = date, ins = ins)
                if not icandle:
                    print('Candle does not exist ' + ins +' '+ str(date))
                    drow[ins+'_vol'] = -999999
                    drow[ins+'_open'] = -999999
                    drow[ins+'_close'] = -999999
                    drow[ins+'_high'] = -999999
                    drow[ins+'_low'] = -999999
                else:
                    drow[ins+'_vol'] = int(icandle['volume'])
                    drow[ins+'_open'] = float(icandle['open'])
                    if float(icandle['close']) > float(icandle['open']):
                        drow[ins+'_close'] = int(1)
                    else:
                        drow[ins+'_close'] = int(-1)
                    drow[ins+'_high'] = float(icandle['high'])-float(icandle['open'])
                    drow[ins+'_low'] = float(icandle['low'])-float(icandle['open'])
            dstr.append(drow)
        df = pd.DataFrame(dstr)
        #code.interact(banner='', local=locals())
        if write_raw:
         print('Constructed DF with shape ' + str(df.shape))
         outname = '/home/ubuntu/data/cexport.csv'
         df.to_csv(outname)
        datecol = df['date'].copy() # copy for usage in improveEstim
        df.drop(['date'],1,inplace = True)
        volp = {}
        for col in df.columns:
         if '_vol' in col or '_open' in col:
          continue
         if improve_model:
          self.improveEstim(col, df, datecol)
         pvol, vprev = self.predictColumn(col, df, newEstim = newEstim)
         parts = col.split('_')
         if len(parts) < 3:
          print('WARNING: Unexpected column ' + col)
          continue
         #print(col)
         instrument = parts[0] + '_' + parts[1]
         typ = parts[2]
         if instrument in volp.keys():
          volp[instrument][typ] = pvol # store diff to prev day
         else:
          volp[instrument] = { typ: pvol }
         print(col + ' ' + str(pvol))
        if write_predict:
         #psort = sorted(volp, key = lambda x: x.get('relative'), reverse = True)
         if complete:
          outfile = open(self.settings['prices_path'],'w')
         else:
          outfile = open(self.settings['prices_path']+'.partial','w') # seperate file for partial estimates
         outfile.write('INSTRUMENT,HIGH,LOW,CLOSE\n')
         for instr in volp.keys():
          outfile.write(str(instr) + ',' + str(volp[instr].get('high')) + ',' + str(volp[instr].get('low')) + ',' + str(volp[instr].get('close')) + '\n')
         outfile.close()
    #def testEstim(self, ins): # this method tests the combined estimators for one instrument
    # dates = []
    # statement = 'select distinct date from dailycandles order by date;'
    # for row in self.db.query(statement):
    #     #if row['date'][:4] == year:
    #     dates.append(row['date'])
    # dates[0]
    def getFeatureImportances(self):
     feature_names = []
     statement = 'select distinct ins from dailycandles order by ins;'
     for row in self.db.query(statement):
         feature_names.append(row['ins'] + '_volume')
         feature_names.append(row['ins'] + '_open')
         feature_names.append(row['ins'] + '_close')
         feature_names.append(row['ins'] + '_high')
         feature_names.append(row['ins'] + '_low')
     sql = 'select distinct name from estimators;'
     for row in self.db.query(sql):
      pcol = row.get('name')
      try:
       dumpname = self.settings.get('estim_path') + pcol
       #regr = pickle.load(open(dumpname,'rb'))
       regr = estim_pipeline(path = dumpname)
      except:
       print('Failed to load model for ' + pcol)
       continue
      print(pcol)
      for name, importance in zip(feature_names, regr.feature_importances_):
       dbline = {'name': pcol, 'feature': name, 'importance': importance}
       self.importances.upsert(dbline, ['name', 'feature'])
    def distToNow(self, idate):
     now = datetime.datetime.now()
     ida = datetime.datetime.strptime(idate,'%Y-%m-%d')
     delta = now - ida
     return math.exp(-delta.days/365.25)# exponentially decaying weight decay
    def improveEstim(self, pcol, df, datecol):
     try:
      dumpname = self.settings.get('estim_path') + pcol
      #regr = pickle.load(open(dumpname,'rb'))
      regr = estim_pipeline(path = dumpname)
     except:
      print('Failed to load model for ' + pcol)
      return
     params = regr.get_params()
     wimper = math.floor(np.random.random()*3)
     n_estimators_base = int(params.get('n_estimators'))
     if wimper == 0:
        n_range = getRangeInt(n_estimators_base,0.1,lower = 10)
     else:
        n_range = [n_estimators_base]
     n_minsample = params.get('min_samples_split')
     if wimper == 1:
        minsample = getRangeInt(n_minsample,0.1,lower = 2)
     else:
        minsample = [n_minsample]
     learning_rate = params.get('learning_rate')
     if wimper == 2:
        n_learn = getRangeFlo(learning_rate,0.01,lower = 0.0001,upper = 1)
     else:
        n_learn = [learning_rate]
     percentile = params.get('percentile')
     # percentile is always considered because this might be the most crucial parameter
     n_perc = getRangeInt(percentile,0.1,1,100)
     parameters = { 'n_estimators': n_range,
                    'min_samples_split': minsample,
                    'learning_rate': n_learn,
                    'percentile': n_perc }    
     weights = np.array(datecol.apply(self.distToNow).values[:])
     x = np.array(df.values[:])
     y = np.array(df[pcol].values[:]) # make a deep copy to prevent data loss in future iterations
     vprev = y[-1]
     weights = weights[1:]
     y = y[1:] # drop first line
     xlast = x[-1,:]
     x = x[:-1,:]# drop the last line
     i = 0
     while i < y.shape[0]:
      if y[i] < -999990: # missing values are marked with -999999
       weights = np.delete(weights, i)
       y = np.delete(y,i)
       x = np.delete(x, (i), axis = 0)
      else:
       i += 1
     if '_close' in pcol:
        base_regr = estim_pipeline(classifier = True)
        #code.interact(banner='', local=locals())
        y = np.array(y,dtype = int).round()
     else:
        base_regr = estim_pipeline()
     score_str = 'neg_mean_absolute_error'
     gridcv = GridSearchCV(base_regr, parameters, cv = 3, iid = False, error_score = 'raise', scoring = score_str) #GradientBoostingRegressor()
     try:
      gridcv.fit(x,y, sample_weight = weights)
     except Exception as e:
      print('FATAL: failed to compute ' + pcol)
      return
     print('Improving Estimator for ' + pcol + ' ' + str(gridcv.best_params_) + ' score: ' + str(gridcv.best_score_))
     #pickle.dump(gridcv.best_estimator_.pipeline, open(dumpname,'wb'))
     gridcv.best_estimator_.writeToDisk(dumpname)
     #joblib.dump(gridcv.best_estimator_, dumpname)
     dobj = {'name': pcol, 'score': gridcv.best_score_ }
     self.estimtable.upsert(dobj, ['name'])
    def predictColumn(self, pcol, df, newEstim = False):
     x = np.array(df.values[:])
     y = np.array(df[pcol].values[:]) # make a deep copy to prevent data loss in future iterations
     vprev = y[-1]
     y = y[1:] # drop first line
     xlast = x[-1,:]
     x = x[:-1,:]# drop the last line
     if newEstim:
      #regr = RandomForestRegressor()
      if '_close' in pcol:
        regr = estim_pipeline(classifier = True) #GradientBoostingRegressor()
      else:
        regr = estim_pipeline() #GradientBoostingRegressor()
      #regr = TPOTRegressor(generations = 50, population_size = 10, verbosity = 2)
      #remove missing lines from the training data
      i = 0
      while i < y.shape[0]:
       if y[i] < -999990: # missing values are marked with -999999
        y = np.delete(y,i)
        x = np.delete(x, (i), axis = 0)
       else:
        i += 1
      regr.fit(x,y)
      dumpname = self.settings.get('estim_path') + pcol
      #joblib.dump(regr, dumpname)
      #pickle.dump(regr, open(dumpname,'wb'))
      regr.writeToDisk(dumpname)    
     else:
      dumpname = self.settings.get('estim_path') + pcol
      #regr = pickle.load(open(dumpname,'rb'))
      regr = estim_pipeline(path = dumpname)
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
     rawUnits = multiplier * targetExp * conversion
     if rawUnits > 0:
      return math.floor(rawUnits)
     else:
      return math.ceil(rawUnits)
     return 0
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
    def openLimit(self, ins, close_only = False, complete = True):
     if complete:
      df = pd.read_csv(self.settings['prices_path'])
     else:
      df = pd.read_csv(self.settings['prices_path']+'.partial')
     op = self.getPrice(ins)
     cl = df[df['INSTRUMENT'] == ins]['CLOSE'].values[0]
     hi = df[df['INSTRUMENT'] == ins]['HIGH'].values[0]+op
     lo = df[df['INSTRUMENT'] == ins]['LOW'].values[0]+op
     price = self.getPrice(ins)
     # get the R2 of the consisting estimators
     colname = ins + '_close'
     row = self.estimtable.find_one(name = colname)
     if row:
      close_score = row.get('score')
     else: 
      print('WARNING: Unscored estimator - ' + colname)
      return None
     colname = ins + '_high'
     row = self.estimtable.find_one(name = colname)
     if row:
      high_score = row.get('score')
     else: 
      print('WARNING: Unscored estimator - ' + colname)
      return None
     colname = ins + '_low'
     row = self.estimtable.find_one(name = colname)
     if row:
      low_score = row.get('score')
     else: 
      print('WARNING: Unscored estimator - ' + colname)
      return None
     #print(ins + ' - cum. R2: ' + str(r2sum))
     spread = self.getSpread(ins)
     trade = None
     for tr in self.trades:
      if tr.instrument == ins:
       trade = tr
     if trade:
      isopen = True
      if close_score < -1:# if we do not trust the estimator we should not move forward
       self.oanda.trade.close(self.settings.get('account_id'), trade.id)
      if trade.currentUnits > 0 and cl < 0:
       self.oanda.trade.close(self.settings.get('account_id'), trade.id)
       isopen = False
      if trade.currentUnits < 0 and cl > 0:
       self.oanda.trade.close(self.settings.get('account_id'), trade.id)
       isopen = False
      if isopen:
       return
     if close_only:
      return # if this flag is set only check for closing and then return
     if close_score < -1:
      return
     if cl > 0:
      step = 1.8*abs(low_score)
      sl = lo - step
      entry = lo
      tp = hi
     else:
      step = 1.8*abs(high_score)
      sl = hi+step
      entry = hi
      tp = lo
     rr = abs((tp-entry)/(sl-entry))
     if rr < 1.5:# Risk-reward too low
      print(ins + ' RR: ' + str(rr) + ' | ' + str(entry) + '/' + str(sl) + '/' + str(tp))
      return None
     # if you made it here its fine, lets open a limit order
     # r2sum is used to scale down the units risked to accomodate the estimator quality
     units = self.getUnits(abs(sl-entry),ins)*min(abs(cl),1.0) # multiply by close value to assign units proportional to certainity
     if units > 0:
      units = math.floor(units)
     if units < 0:
      units = math.ceil(units)
     if abs(units) < 1:
      return None # oops, risk threshold too small
     if tp < sl:
      units *= -1
     pipLoc = self.getPipSize(ins)
     if abs(sl-entry) < 200*10**(-pipLoc): # sl too small
      return None
     if (entry-price)*units > 0:
      otype = 'STOP'
     else:
      otype = 'LIMIT'
     fstr = '30.' + str(pipLoc) + 'f'
     tp = format(tp, fstr).strip()
     sl = format(sl, fstr).strip()
     entry = format(entry, fstr).strip()
     expiry = datetime.datetime.now() + datetime.timedelta(days=1)
     args = {'order': {
     'instrument': ins,
     'units': units,
     'price': entry,
     'type': otype,
     'timeInForce': 'GTD',
     'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
     'takeProfitOnFill': {'price': tp, 'timeInForce': 'GTC'},
     'stopLossOnFill': {'price': sl, 'timeInForce': 'GTC'},
     }}
     #code.interact(banner='', local=locals())
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
