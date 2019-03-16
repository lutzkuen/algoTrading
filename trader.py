from observer import controller_cython as controller
from observer_keras import controller_cython as controller_keras
cont = controller.Controller('/home/tubuntu/settings_triangle.conf', 'live')
allowed_ins = [ins.name for ins in cont.allowed_ins]
returns = [cont.open_limit(ins, duration=12, split_position=False, adjust_rr=True) for ins in allowed_ins]
cont_demo = controller_keras.Controller('/home/tubuntu/settings_triangle.conf', 'demo')
#returns = [cont_demo.simplified_trader(ins) for ins in allowed_ins]
returns = [cont_demo.open_limit(ins, duration=12, split_position=False, adjust_rr=True) for ins in allowed_ins]
returns = [cont_demo.simplified_trader(ins) for ins in ['USD_JPY', 'USD_CHF', 'EUR_USD', 'GBP_USD']]
