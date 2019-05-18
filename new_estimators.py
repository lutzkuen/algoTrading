import datetime
try:
    from observer import controller_cython as controller
    # from observer import controller as controller
except ImportError:
    print('WARNING: Cython module not found, falling back to native python')
    from observer import controller as controller
cont = controller.Controller('../settings_triangle.conf', 'live', verbose=1)
now = datetime.datetime.now() - datetime.timedelta(hours=24)
now = now.strftime('%Y-%m-%d')
cont.data2sheet(read_raw=True, write_raw=True, improve_model=True, maxdate=now, complete=False, append_raw=True)
