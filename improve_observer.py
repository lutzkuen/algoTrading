import time
import datetime
try:
    from observer import controller_cython as controller
    #from observer import controller as controller
except ImportError:
    print('WARNING: Cython module not found, falling back to native python')
    from observer import controller as controller
while True:
    sleep_time = 60*10 # sleep for 10 min if credit count available
    try:
        cont = controller.Controller('/home/tubuntu/settings_triangle.conf', 'live', verbose=0)
        cont.data2sheet(improve_model=True, write_predict=False, write_raw=True, read_raw=False, append_raw=True)
    except Exception as e:
        print(e)
    time.sleep(sleep_time)
