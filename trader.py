from observer import controller_cython as controller
cont = controller.Controller('/home/tubuntu/settings_triangle.conf', 'live')
allowed_ins = [ins.name for ins in cont.allowed_ins]
returns = [cont.open_limit(ins, duration=12) for ins in allowed_ins]
cont_demo = controller.Controller('/home/tubuntu/settings_triangle.conf', 'demo', write_trades=True, multiplier=10)
returns = [cont_demo.open_limit(ins, duration=16) for ins in allowed_ins]
