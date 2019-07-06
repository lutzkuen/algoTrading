# from swingAssistant import controller_cython as controller
from observer import controller_cython as cobs
print('live')
# cont = controller.Controller('/home/tubuntu/settings_swing.conf', 'live')
# cont.reduce_exposure()
oco = cobs.Controller('/home/tubuntu/settings_triangle.conf', 'live')
oco.move_stops()
print('demo')
# cont_demo = controller.Controller('/home/tubuntu/settings_swing.conf', 'demo')
# cont_demo.reduce_exposure()
ocodemo = cobs.Controller('/home/tubuntu/settings_triangle.conf', 'demo')
ocodemo.move_stops()
