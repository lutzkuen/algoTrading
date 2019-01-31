from observer import controller_cython as controller
cont = controller.Controller('/home/ubuntu/settings_triangle.conf','demo')
cont.reduce_risk()
cont = controller.Controller('/home/ubuntu/settings_triangle.conf','live')
cont.reduce_risk()
