try:
    #from observer import controller_cython as controller
    from observer import controller as controller
except ImportError:
    print('WARNING: Cython module not found, falling back to native python')
    from observer import controller as controller
cont = controller.Controller('../settings_triangle.conf', None, verbose=1)
cont.data2sheet(new_estim=True, write_predict=False, write_raw=False)
