# read in the data file 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def predictVolume(pcol, df):
 x = np.array(df.values[:])
 y = np.array(df[pcol].values[:]) # make a deep copy to prevent data loss in future iterations
 vmean = np.mean(y)
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
 return yp[0], vmean

fname = '../../data/cexport.csv'
df = pd.read_csv(fname)
df.drop(['date'],1,inplace = True)
volp = []
for col in df.columns:
 if '_vol' in col:
  pvol, vmean = predictVolume(col, df)
  nv = { 'ins': col, 'predict': pvol, 'mean': vmean, 'relative': pvol/vmean }
  volp.append(nv)
  print(col + ' ' + str(pvol))

psort = sorted(volp, key = lambda x: x.get('relative'), reverse = True)
outfile = open('volume.csv','w')
outfile.write('INSTRUMENT,PREDICTION,MEAN,RELATIVE\n')
for nv in psort:
 outfile.write(str(nv.get('ins')) + ',' + str(nv.get('predict')) + ',' + str(nv.get('mean')) + ',' + str(nv.get('relative')) + '\n')
outfile.close()