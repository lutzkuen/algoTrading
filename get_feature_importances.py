from observer import controller_cython as controller
cont = controller.Controller('/home/ubuntu/settings_triangle.conf',None)
cont.get_feature_importances()
