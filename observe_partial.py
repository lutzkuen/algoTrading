from observer import controller
cont = controller.controller('/home/ubuntu/settings_triangle.conf','live')
cont.retrieveData(4, completed = False, upsert = True)
cont.data2sheet(complete = False)
allowed_ins = [ins.name for ins in cont.allowed_ins]
returns = [cont.openLimit(ins, close_only = True, complete = False) for ins in allowed_ins]
cont_demo = controller.controller('/home/ubuntu/settings_triangle.conf','demo')
returns = [cont_demo.openLimit(ins, close_only = True, complete = False) for ins in allowed_ins]
