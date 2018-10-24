try:
    from observer import controller_cython as controller
except ImportError:
    print('WARNING: Cython module not found, falling back to native python')
    from observer import controller as controller
cont = controller.Controller('../settings_triangle.conf', None)
cont.get_feature_importances()
