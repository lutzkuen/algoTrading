from observer import controller_cython as controller
cont = controller.Controller('../settings_triangle.conf',None)
cont.data2sheet(new_estim = True, write_predict = False, write_raw = False)
