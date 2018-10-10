# read in the data file 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def predictColumn(pcol, df):
 x = np.array(df.values[:])
 y = np.array(df[pcol].values[:]) # make a deep copy to prevent data loss in future iterations
 vprev = y[-1]
 y = y[1:] # drop first line
 xlast = x[-1,:]
 x = x[:-1,:]# drop the last line
 regr = RandomForestRegressor(n_estimators = 100)
 #parameters = { 'n_estimators': range(10,100,10) }
 #regr = GridSearchCV(RandomForestRegressor(), parameters)
 
 #remove missing lines from the training data
 i = 0
 while i < y.shape[0]:
  if y[i] < 0: # missing values are marked with -1
   y = np.delete(y,i)
   x = np.delete(x, (i), axis = 0)
  else:
   i += 1
 regr.fit(x,y)
 #print('Using: ' + str(regr.best_params_))
 yp = regr.predict(xlast.reshape(1, -1))
 return yp[0], vprev

fname = '../../data/cexport.csv'
df = pd.read_csv(fname)
df.drop(['date'],1,inplace = True)
df.drop(['Unnamed: 0'],1,inplace = True)# i do not know where this weird index like column comes from. It does only occur if we save to file and reload. Probably the index is saved without col label
volp = {}
for col in df.columns:
 pvol, vprev = predictColumn(col, df)
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
outfile = open('prices.csv','w')
outfile.write('INSTRUMENT,HIGH,LOW,OPEN,CLOSE,VOLUME\n')
for instr in volp.keys():
 outfile.write(str(instr) + ',' + str(volp[instr].get('high')) + ',' + str(volp[instr].get('low')) + ',' + str(volp[instr].get('open')) + ',' + str(volp[instr].get('close')) + ',' + str(volp[instr].get('vol')) + '\n')
outfile.close()