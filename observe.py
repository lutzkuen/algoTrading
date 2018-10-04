from observer import controller
cont = controller.controller('/home/ubuntu/settings_triangle.conf','live')
cont.retrieveData(4)
cont.data2sheet()
allowed_ins = [ins.name for ins in cont.allowed_ins]
#returns = [cont.openLimit(ins) for ins in allowed_ins]
cont_demo = controller.controller('/home/ubuntu/settings_triangle.conf','demo')
returns = [cont_demo.openLimit(ins) for ins in allowed_ins]
