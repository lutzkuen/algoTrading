#!/bin/bash
/usr/bin/python3 /home/ubuntu/algoTrading/utils/forexfactory_webscraper.py delta_load /home/ubuntu/settings_triangle.conf
/usr/bin/python3 /home/ubuntu/algoTrading/utils/myfxbook_positionbook.py /home/ubuntu/settings_triangle.conf
/usr/bin/python3 /home/ubuntu/algoTrading/observe.py
/usr/bin/python3 /home/ubuntu/algoTrading/utils/myfxbook_positionbook.py /home/ubuntu/settings_triangle.conf
