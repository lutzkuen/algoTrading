#from observer import controller as controller
from observer import controller_cython as controller
cont = controller.Controller('/home/tubuntu/settings_triangle.conf','live')
cont.retrieve_data(1, upsert = True)
cont.retrieve_data(1,completed = False, upsert = True)
cont.predict_tomorrow()
cont_demo = controller.Controller('/home/tubuntu/settings_triangle.conf','demo')
allowed_ins = [ins.name for ins in cont_demo.allowed_ins]
returns = [cont_demo.open_limit(ins, duration=12) for ins in allowed_ins]
#returns = [cont_demo.simplified_trader(ins) for ins in allowed_ins]
#returns = [cont_demo.open_limit(ins, close_only = True, complete = False) for ins in allowed_ins]
