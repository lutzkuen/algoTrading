from observer import controller_cython as controller
cont = controller.controller('/home/ubuntu/settings_triangle.conf','live')
allowed_ins = [ins.name for ins in cont.allowed_ins]
returns = [cont.openLimit(ins, close_only = True) for ins in allowed_ins]
cont_demo = controller.controller('/home/ubuntu/settings_triangle.conf','demo')
returns = [cont_demo.openLimit(ins, close_only = True) for ins in allowed_ins]
