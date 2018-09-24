from observer import controller
cont = controller.controller('/home/ubuntu/settings_triangle.conf','live')
cont.retrieveData(2)
cont.data2sheet()
cont_demo = controller.controller('/home/ubuntu/settings_triangle.conf','demo')
allowed_ins = [ins.name for ins in cont_demo.allowed_ins]
returns = [cont_demo.openLimit(ins) for ins in allowed_ins]
