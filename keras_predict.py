import time
import datetime
from observer_keras import controller as controller

cont = controller.Controller('../../Documents/settings_triangle.conf', 'demo', verbose=1)
cont.get_latest_prediction()