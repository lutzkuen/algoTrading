#!/usr/bin/env python
__author__ = "Lutz Künneke"
__copyright__ = ""
__credits__ = [ '@SignalFactory' ]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Lutz Künneke"
__email__ = "lutz.kuenneke89@gmail.com"
__status__ = "Prototype"
"""
follow fx signals supplied by @SignalFactory
Use at own risk
Author: Lutz Kuenneke, 11.07.2018
"""
import tweepy
from tweepy import Stream
from tweepy.auth import OAuthHandler
import json
import configparser
import v20
import datetime
#from tweepy import auth

class oandaWrapper(object):
 def __init__(self):
  confname = '../settings_v20.conf'
  config = configparser.ConfigParser()
  config.read(confname)
  self.settings = {}
  self.demo = {}
  self.demo['domain'] = config.get('demo', 'streaming_hostname')
  self.demo['access_token'] = config.get('demo', 'token')
  self.demo['account_id'] = config.get('demo', 'active_account')
  self.demo['v20_host'] = config.get('demo', 'hostname')
  self.demo['v20_port'] = config.get('demo', 'port')
  self.demo_oanda = v20.Context(
      self.demo.get('v20_host'),
      port=self.demo.get('v20_port'),
      token=self.demo.get('access_token'))
 def openTrade(self, istr):
  # find out whether it is an open
  sarr = istr.split('|')
  if not sarr[0] == 'Forex Signal ':
   print('First check failed')
   return
  if 'Close' in sarr[1]:
   print('Just a close info')
   return
  if 'Buy' in sarr[1]:
   direction = 1
  if 'Sell' in sarr[1]:
   direction = -1
  insarr = sarr[1].split('@')
  _ins = insarr[0][-6:]
  ins = _ins[:3] + '_' + _ins[3:]
  entry = insarr[1].strip()
  for sar in sarr:
   if 'TP:' in sar:
    _ispl = sar.split(':')
    take_profit = _ispl[1].strip()
   if 'SL:' in sar:
    _ispl = sar.split(':')
    stop_loss = _ispl[1].strip()
  units = int(direction*100)
  expiry = datetime.datetime.now() + datetime.timedelta(hours = 1)
  print('Opening ' + str(units) + ' on ' + ins)
  args = {
            'order': {
                'instrument': ins,
                'units': units,
                'type': 'LIMIT',
                'price': entry,
                'timeInForce': 'GTD',
                'gtdTime': expiry.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'takeProfitOnFill': {
                    'price': take_profit,
                    'timeInForce': 'GTC'},
                'stopLossOnFill': {
                    'price': stop_loss,
                    'timeInForce': 'GTC'
                    }}}
  #print(args)
  ticket = self.demo_oanda.order.create(self.demo.get('account_id'), **args)
  #ticket_json = json.loads(ticket.raw_body)
  #print(ticket_json)
class MyListener(object):
    def __init__(self):
        self.oanda = oandaWrapper()
    def on_data(self, data):
        try:
            #with open('python.json', 'a') as f:
            #    f.write(data)
            #    return True
            tjson = json.loads(data)
            if tjson.get('user').get('screen_name') == 'SignalFactory':
                print(tjson.get('text') + ' | ' + tjson.get('user').get('screen_name'))
                self.oanda.openTrade(tjson.get('text'))
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
    def keep_alive(self):
        pass
    def on_error(self, status):
        print('Error ' + str(status))
        return True
    def on_exception(self, exception):
        print('Exception ' + str(exception))
        return True
    def on_connect(self):
        print('Connecting')
        return True
if __name__ == '__main__': 
 #oanda = oandaWrapper()
 confname = '../tweepy.conf'
 config = configparser.ConfigParser()
 config.read(confname)
 consumer_key = config.get('tweepy','consumer_key')
 consumer_secret = config.get('tweepy','consumer_secret')
 access_token = config.get('tweepy','access_token')
 access_secret = config.get('tweepy','access_secret')
  
 auth = OAuthHandler(consumer_key, consumer_secret)
 auth.set_access_token(access_token, access_secret)
  
 api = tweepy.API(auth)
 
  
 twitter_stream = Stream(auth, MyListener())
 #twitter_stream.filter(track=['forex,fx'])
 twitter_stream.filter(track=['forex'])#, follow=['SignalFactory'])
