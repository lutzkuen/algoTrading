from observer import controller_cython as controller
cont = controller.Controller('/home/ubuntu/settings_triangle.conf','live')
cont.retrieve_data(4, upsert = True)
cont.data2sheet()
