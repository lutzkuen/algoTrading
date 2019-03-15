import v20
import time
import configparser
import numpy as np
import datetime


def checkaccount(settings_path, acctype):
    config = configparser.ConfigParser()
    config.read(settings_path)
    settings = dict()
    settings['domain'] = config.get(acctype, 'streaming_hostname')
    settings['access_token'] = config.get(acctype, 'token')
    settings['account_id'] = config.get(acctype, 'active_account')
    settings['host'] = config.get(acctype, 'hostname')
    settings['port'] = config.get(acctype, 'port')
    settings['account_risk'] = float(config.get('triangle', 'account_risk'))
    settings['account_target'] = float(config.get('triangle', 'account_target'))
    threshold = settings['account_risk'] * settings['account_target']
    if acctype == 'demo':
        threshold *= 10
    oanda = v20.Context(settings.get('host'), port=settings.get('port'), token=settings.get('access_token'))
    trades = oanda.trade.list_open(settings.get('account_id')).get('trades', '200')
    unrealizedPL = 0
    for trade in trades:
        unrealizedPL += float(trade.unrealizedPL)
    print(str(datetime.datetime.now()) + ': ' + acctype + ' uPL: ' + str(unrealizedPL) + ' limit ' + str(threshold))
    if abs(unrealizedPL) >= threshold:
        # close all trades and orders
        for trade in trades:
            response = oanda.trade.close(settings.get('account_id'), trade.id)
            print(response.raw_body)
        # close the orders too
        #orders = oanda.order.list(settings.get('account_id')).get('orders', '200')
        #for order in orders:
        #    response = oanda.order.close(settings.get('account_od'), order.id)
        #    print(response.raw_body)

if __name__ == '__main__':
    while True:
        try:
            checkaccount('/home/tubuntu/settings_triangle.conf', 'demo')
        except Exception as e:
            print(str(e))
        time.sleep(60*15)
