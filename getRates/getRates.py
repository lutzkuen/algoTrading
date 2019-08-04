import os
import json
import wget
import code
import numpy as np
import configparser
import v20


class RatesController(object):

    def __init__(self, config_name):
        self.config = configparser.ConfigParser()
        self.prices = dict()
        self.config.read(config_name)
        self.fxmath = self.get_fxmath()
        self.oanda = v20.Context(self.config.get('live', 'hostname'),
                                 port=self.config.get('live', 'port'),
                                 token=self.config.get('live', 'token'
                                                       ))
        self.allowed_ins = \
            self.oanda.account.instruments(self.config.get('live', 'active_account'
                                                           )).get('instruments', '200')
        self.trades = self.oanda.trade.list_open(self.config.get('live', 'active_account')).get('trades', '200')

    def get_price(self, ins):
        """
        Returns price for a instrument
        ins: Instrument, e.g. EUR_USD
        """
        args = {'instruments': ins}
        if ins in self.prices.keys():
            return self.prices[ins]
        price_raw = self.oanda.pricing.get(self.config.get('live', 'active_account'
                                                           ), **args)
        price_json = json.loads(price_raw.raw_body)
        price = (float(price_json.get('prices')[0].get('bids')[0].get('price'
                                                                      )) + float(
            price_json.get('prices')[0].get('asks'
                                            )[
                0].get(
                'price'))) / 2.0
        self.prices[ins] = price
        return price

    def get_conversion(self, leading_currency):
        """
        get conversion rate to account currency

        :param leading_currency: ISO Code of the leading currency for the traded pair
        :return:
        """

        account_currency = 'EUR'
        # trivial case
        if leading_currency == account_currency:
            return 1
        # try direct conversion
        for ins in self.allowed_ins:
            if leading_currency in ins.name and account_currency in ins.name:
                price = self.get_price(ins.name)
                if ins.name.split('_')[0] == account_currency:
                    return price
                else:
                    return 1.0 / price
        # try conversion via usd
        eurusd = self.get_price('EUR_USD')
        if not eurusd:
            return None
        for ins in self.allowed_ins:
            if leading_currency in ins.name and 'USD' in ins.name:
                price = self.get_price(ins.name)
                if not price:
                    return None
                if ins.name.split('_')[0] == 'USD':
                    return price / eurusd
                else:
                    return 1.0 / (price * eurusd)
        return None

    def get_fxmath(self):
        fxmath_url = self.config.get('data', 'fxmath_url')
        fxmath_filename = wget.download(fxmath_url)
        f = open(fxmath_filename)
        # code.interact(banner='', local=locals())
        fxmath = dict()
        # arrname = ''
        for line in f.readlines():
            # code.interact(banner='', local=locals())
            arrname, vals = line.split('=new ')
            # if '=new' in line:
            arrname = arrname.replace('var ', '').replace('=new', '').replace('\n', '').replace('\r', '')
            # else:
            vals = vals.replace('\n', '').replace('Array(', '').replace(');', '').replace('"', '').split(',')
            fxmath[arrname] = vals
            try:
                fxmath[arrname] = np.array(fxmath[arrname]).astype(np.float32)
            except:
                pass
        # clean up after yourself
        f.close()
        os.remove(fxmath_filename)
        return fxmath

    def check_trades(self):
        """
        Iterates over open trades and prints their interest earned over 24 H
        :return: dict with trades and 24h interest
        """
        for trade in self.trades:
            if trade.currentUnits > 0:
                side = 'long'
            else:
                side = 'short'
            interest = self.get_interest(trade.instrument, side, units=abs(trade.currentUnits), verbose=True)
            if interest < 0:
                print('ATTENTION: You are holding a negative carry!')

    def check_cfds(self, target_units=1000):
        """
        Check all available CFDs for carry trade opportunitys
        :param target_units: Units in EUR
        :return: array of positive carry trade opps
        """
        # code.interact(banner='', local=locals())
        carry_trades = []
        for instrument in self.allowed_ins:
            if not instrument.type == 'CFD':
                continue
            base, quote = instrument.name.split('_')
            conversion = self.get_conversion(base)
            if not conversion:
                price = self.get_price(instrument.name)
                conversion = self.get_conversion(quote) / price
                if not conversion:
                    print('WARNING: Could not convert ' + str(instrument.name))
                    continue
            # check the long side
            # print('Checking ' + instrument.name)
            units = target_units * conversion
            interest = self.get_interest(instrument.name, 'long', units=units)
            if interest > 0:
                carry_trades.append(
                    {'instrument': instrument.name, 'side': 'long', 'units': units, 'interest': interest})
            interest = self.get_interest(instrument.name, 'short', units=units)
            if interest > 0:
                carry_trades.append(
                    {'instrument': instrument.name, 'side': 'short', 'units': units, 'interest': interest})
        carry_trades = sorted(carry_trades, key=lambda x: x.get('interest') ,reverse=True)
        return carry_trades

    def get_interest(self, instrument, side, hours=24.0, units=1000, verbose=False):
        base_ins, quote_ins = instrument.split('_')
        duration = hours / 8766.0
        idx_base = self.fxmath['arrInterestCode'].index(base_ins)
        idx_quote = self.fxmath['arrInterestCode'].index(quote_ins)
        price = self.get_price(instrument)
        conversion_base = self.get_conversion(base_ins)
        # code.interact(banner='', local=locals())
        if not conversion_base:
            conversion_base = self.get_conversion(quote_ins)
            conversion_base = price / conversion_base
        conversion_quote = self.get_conversion(quote_ins)
        # code.interact(banner='', local=locals())
        if not conversion_quote:
            conversion_quote = self.get_conversion(base_ins)
            conversion_quote = price / conversion_quote
        if side == 'long':
            base = units * (self.fxmath['arrInterestBorrow'][idx_base] / 100.0) * duration / conversion_base
            quote = units * price * (self.fxmath['arrInterestLend'][idx_quote] / 100.0) * duration / conversion_quote
            interest = base - quote
        else:
            base = units * (self.fxmath['arrInterestLend'][idx_base] / 100.0) * duration / conversion_base
            quote = units * price * (self.fxmath['arrInterestBorrow'][idx_quote] / 100) * duration / conversion_quote
            interest = quote - base
        if verbose:
            print(instrument + ' ' + side + ' ' + str(units) + ' ' + str(interest))
        return interest
