from observer import controller_cython as controller
cont = controller.controller('/home/ubuntu/settings_triangle.conf',None)
cont.getFeatureImportances()
