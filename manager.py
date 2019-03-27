from observer import controller_cython as controller
#cont = controller.Controller('/home/tubuntu/settings_triangle.conf', 'live')
cont_demo = controller.Controller('/home/tubuntu/settings_triangle.conf', 'demo')
cont_demo.manage_portfolio()
