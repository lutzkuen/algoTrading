#!/bin/bash
/usr/bin/python3 /home/tubuntu/algoTrading/utils/forexfactory_webscraper.py delta_load /home/tubuntu/settings_triangle.conf
/usr/bin/python3 /home/tubuntu/algoTrading/observe.py
/usr/bin/python3 /home/tubuntu/algoTrading/utils/myfxbook_positionbook.py /home/tubuntu/settings_triangle.conf
