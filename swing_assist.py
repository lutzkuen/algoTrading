from swingAssistant import controller_cython as controller

cont = controller.Controller('/home/tubuntu/settings_swing.conf', 'demo')
cont.check_positions()
