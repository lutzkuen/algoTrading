try:
    from observer import controller_cython as controller
    #from observer import controller as controller
except ImportError:
    print('WARNING: Cython module not found, falling back to native python')
    from observer import controller as controller

cont = controller.Controller('../settings_triangle.conf', None, verbose=2)
cont.data2sheet(improve_model=True, write_predict=False, read_raw=True)
