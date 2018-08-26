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
Backtest for algoTrader
Use at own risk
Author: Lutz Kuenneke, 12.07.2018
"""

from algoTrader.backtest import controller
cont = controller('/home/ubuntu/settings_triangle.conf','backtest')
cont.runBacktest()
