from observer import controller_cython as controller
cont = controller.Controller('/home/ubuntu/settings_triangle.conf','demo')
cont.reduce_risk()
