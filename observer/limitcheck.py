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
    oanda = v20.Context(settings.get('host'), port=settings.get('port'), token=settings.get('access_token'))
    unrealizedPL = 0
    try:
        now = str(datetime.datetime.now())
        trades = oanda.trade.list_open(settings.get('account_id')).get('trades', '200')
        for trade in trades:
            unrealizedPL += float(trade.unrealizedPL)
        treshold = -1
        if unrealizedPL > threshold:
            for trade in trades:
                response = oanda.trade.close(settings.get('account_id'), trade.id)
            print(response)
            orders = oanda.order.list_pending(settings.get('account_id')).get('orders', '200')
            for order in orders:
                response = oanda.order.cancel(settings.get('account_id'), order.id)
        else:
            print(now +' '+str(unrealizedPL) + ' - ' + str(threshold))

    except Exception as e:
        print(str(e))
        time.sleep(5)

if __name__ == '__main__':
    while True:
        checkaccount('/home/tubuntu/settings_triangle.conf', 'demo')
