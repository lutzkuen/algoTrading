from swingAssistant import controller_cython as controller
print('live')
cont = controller.Controller('/home/tubuntu/settings_swing.conf', 'live')
cont.reduce_exposure()
print('demo')
cont_demo = controller.Controller('/home/tubuntu/settings_swing.conf', 'demo')
cont_demo.reduce_exposure()
