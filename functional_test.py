#from observer import controller as controller
from observer import controller_cython as controller
cont = controller.Controller('/home/tubuntu/settings_triangle.conf','live')
cont.retrieve_data(1, upsert = True)
cont.retrieve_data(1,completed = False, upsert = True)
cont.data2sheet()
allowed_ins = [ins.name for ins in cont.allowed_ins]
<<<<<<< HEAD
cont_demo = controller.Controller('/home/tubuntu/settings_triangle.conf','demo')
=======
cont_demo = controller.Controller('/home/ubuntu/settings_triangle.conf', 'demo', write_trades=True)
>>>>>>> 637c5e72397f70b46da8fe69bbb943d236204696
returns = [cont_demo.open_limit(ins) for ins in allowed_ins]
returns = [cont_demo.open_limit(ins, close_only = True, complete = False) for ins in allowed_ins]
