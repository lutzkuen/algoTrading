import time
import datetime
from observer_keras import controller_cython as controller

cont = controller.Controller('../settings_triangle.conf', 'demo', verbose=1)
cont.improve_estimator()
