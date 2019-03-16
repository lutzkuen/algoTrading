import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def prevDay(date):
 spl = date.split('-')
 yea = spl[0]
 mon = spl[1]
 day = int(spl[2])
 if day > 10:
  day -=1
  return str(yea)+'-'+str(mon)+'-'+str(day)
 if day > 1:
  day -=1
  return str(yea)+'-'+str(mon)+'-0'+str(day)
 if int(mon) > 10:
  day =31
  mon = int(mon)-1
  return str(yea)+'-'+str(mon)+'-'+str(day)
 if int(mon) > 1:
  day =31
  mon = int(mon)-1
  return str(yea)+'-0'+str(mon)+'-'+str(day)
 return None

fname ='/home/ubuntu/data/cexport_2018.csv'

df = pd.read_csv(fname)

x =[]
y =[]

for date in df['DATE'].values:
 print('preparing '+date)
 target = df[df['DATE']==date]['EUR_USD_vol'].values[0]
 prev = prevDay(date)
 if not prev:
  continue
 y.append(target)
 x.append(df[df['DATE']==date].values[:])

reg = RandomForestRegressor()

reg.fit(x,y)

yp = reg.predict(y)

mse = np.sqrt(np.mean((np.array(y)-np.array(yp))))

print('mse: '+str(mse))
