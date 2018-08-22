#!/usr/bin/env python
__author__ = "Lutz Künneke"
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Lutz Künneke"
__email__ = "lutz.kuenneke89@gmail.com"
__status__ = "Prototype"
"""
powers up the algo Trader controller to run on all allowed demo instruments
https://forums.babypips.com/t/the-3-ducks-trading-system/6430
Use at own risk
Author: Lutz Kuenneke, 12.07.2018
"""

from algoTrader.controller import controller
cont = controller('/home/ubuntu/settings_triangle.conf','demo')
allowed_ins = [ins.name for ins in cont.allowed_ins]
returns = [cont.checkIns(ins) for ins in allowed_ins]
