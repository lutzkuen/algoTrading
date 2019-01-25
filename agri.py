#from agriculture import controller_cython as controller
from agricultural import controller
cont_demo = controller.Controller('/home/ubuntu/settings_triangle.conf','demo')
cont_demo.open_orders()
