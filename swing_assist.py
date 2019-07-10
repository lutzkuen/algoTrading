from observer import controller_cython as cobs
print('live')
oco = cobs.Controller('/home/tubuntu/settings_triangle.conf', 'live')
oco.move_stops()
print('demo')
ocodemo = cobs.Controller('/home/tubuntu/settings_triangle.conf', 'demo')
ocodemo.move_stops()
