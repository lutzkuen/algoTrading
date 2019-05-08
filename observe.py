from observer import controller_cython as controller
#from observer_keras import controller_cython as controller_keras

settings_file = '/home/tubuntu/settings_triangle.conf'
cont = controller.Controller(settings_file, 'live')
cont.retrieve_data(4, completed=False, upsert=True)
cont.predict_tomorrow()
