from observer import controller as controller
#from observer_keras import controller_cython as controller_keras

settings_file = '../settings_triangle.conf'
cont = controller.Controller(settings_file, 'live')
# cont.retrieve_data(4, completed=False, upsert=True)
# cont.data2sheet()
cont.predict_tomorrow()
#cont_keras = controller_keras.Controller(settings_file, 'live')
#cont_keras.get_latest_prediction()
