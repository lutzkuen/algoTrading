from observer import controller_cython as controller
cont = controller.Controller('/home/ubuntu/settings_triangle.conf','live')
allowed_ins = [ins.name for ins in cont.allowed_ins]
returns = [cont.open_limit(ins, close_only = True) for ins in allowed_ins]
cont_demo = controller.Controller('/home/ubuntu/settings_triangle.conf','demo')
returns = [cont_demo.open_limit(ins, close_only = True) for ins in allowed_ins]
