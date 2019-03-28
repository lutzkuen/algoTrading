from observer import controller_cython as controller
import datetime
now = datetime.datetime.now()
if now.hour > 12:
    close_only = True
else:
    close_only = False
# cont = controller.Controller('/home/tubuntu/settings_triangle.conf', 'live')
# cont_demo = controller.Controller('/home/tubuntu/settings_triangle.conf', 'live')
# cont_demo.manage_portfolio(close_only=close_only)
cont_demo = controller.Controller('/home/tubuntu/settings_triangle.conf', 'demo')
cont_demo.manage_portfolio(close_only=close_only)
