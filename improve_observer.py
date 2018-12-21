import time
import datetime
from utils.get_credits import get_current_credits
try:
    from observer import controller_cython as controller
    #from observer import controller as controller
except ImportError:
    print('WARNING: Cython module not found, falling back to native python')
    from observer import controller as controller
while True:
    credits = get_current_credits()
    print(str(datetime.datetime.now()) + ' Current CPU Credits: ' + str(credits))
    if credits >= 140:
        cont = controller.Controller('/home/ubuntu/settings_triangle.conf', 'live', verbose=0)
        cont.data2sheet(improve_model=True, write_predict=False, write_raw=True, read_raw=False, append_raw=True)
    time.sleep(600)
