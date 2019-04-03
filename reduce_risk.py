from observer import controller_cython as controller
cont = controller.Controller('/home/tubuntu/settings_triangle.conf','live')
cont.check_end_of_day()
