from keepLimit import controller
import pandas as pd

cont = controller.Controller('/home/tubuntu/settings_triangle.conf', 'live')
df = pd.read_csv('/home/tubuntu/orderlist.csv')

for i, row in df.iterrows():
    cont.open_limit(row)
