from observer import controller_cython as controller
cont = controller.Controller('/home/ubuntu/settings_triangle.conf', 'live', write_trades=True)
allowed_ins = [ins.name for ins in cont.allowed_ins]
returns = [cont.open_limit(ins) for ins in allowed_ins]
cont_demo = controller.Controller('/home/ubuntu/settings_triangle.conf','demo')
returns = [cont_demo.open_limit(ins) for ins in allowed_ins]
