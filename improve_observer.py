from observer import controller_cython as controller
#cimport observer.controller
cont = controller.controller('../settings_triangle.conf',None)
cont.data2sheet(improve_model = True, write_predict = False, write_raw = False)
