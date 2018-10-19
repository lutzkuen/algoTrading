try:
    from observer import controller_cython as controller
except ImportError:
    print('WARNING: Cython module not found, falling back to native python')
    from observer import controller as controller
#cimport observer.controller
cont = controller.Controller('../settings_triangle.conf',None)
cont.data2sheet(improve_model = True, write_predict = False, write_raw = False)
