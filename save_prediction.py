from observer import controller_cython as controller
import datetime
cont = controller.Controller('/home/tubuntu/settings_triangle.conf','demo')
now = datetime.datetime.now()
yesterday = now - datetime.timedelta(hours=24)
cont.save_prediction_to_db(yesterday.strftime('%Y-%m-%d'))
