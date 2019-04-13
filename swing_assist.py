from swingAssistant import controller_cython as controller

cont = controller.Controller('/home/tubuntu/settings_swing.conf', 'live')
cont.reduce_exposure()
cont_demo = controller.Controller('/home/tubuntu/settings_swing.conf', 'demo')
cont_demo.reduce_exposure()
