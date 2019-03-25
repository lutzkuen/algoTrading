#!/bin/bash
/usr/bin/python3 /home/tubuntu/algoTrading/utils/forexfactory_webscraper.py delta_load /home/tubuntu/settings_triangle.conf
/usr/bin/python3 /home/tubuntu/algoTrading/observe_partial.py
cp /home/tubuntu/data/prices.csv /home/tubuntu/data/prices.csv.backup
cp /home/tubuntu/data/prices.csv.partial /home/tubuntu/data/prices.csv
