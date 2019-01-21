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
    credits = get_current_credits('i-08aeadbe49a9cd6e5')
    print(str(datetime.datetime.now()) + ' Current CPU Credits: ' + str(credits))
    if not credits:
        credits = 145
        sleep_time = 60*60 # sleep for an hour if no credit count available
    else:
        sleep_time = 60*10 # sleep for 10 min if credit count available
    if credits >= 140:
        try:
            cont = controller.Controller('/home/ubuntu/settings_triangle.conf', 'live', verbose=0)
            cont.data2sheet(improve_model=True, write_predict=False, write_raw=True, read_raw=False, append_raw=True)
        except Exception as e:
            print(e)
    time.sleep(sleep_time)
