#!/usr/bin/env python
__author__ = "Lutz Künneke"
__copyright__ = ""
__credits__ = ['@SignalFactory']
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Lutz Künneke"
__email__ = "lutz.kuenneke89@gmail.com"
__status__ = "Prototype"
"""
Publish my mechanical Signals via twitter
Author: Lutz Kuenneke, 28.02.2019
"""
import tweepy
from tweepy import Stream
from tweepy.auth import OAuthHandler
import json
import configparser
import datetime
import pandas as pd
#from tweepy import auth

def shorten(inword, ilen=5):
    len_total = len(str(inword))
    len_before = len(str(int(round(float(inword)))))
    len_round = ilen - len_before
    return round(float(inword), len_round)

if __name__ == '__main__':
    #oanda = oandaWrapper()
    confname = '/home/tubuntu/tweepy.conf'
    config = configparser.ConfigParser()
    config.read(confname)
    consumer_key = config.get('tweepy', 'consumer_key')
    consumer_secret = config.get('tweepy', 'consumer_secret')
    access_token = config.get('tweepy', 'access_token')
    access_secret = config.get('tweepy', 'access_secret')

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth)

    price_list = '/home/tubuntu/data/majors.csv'
    majors = ['EUR_USD', 'USD_JPY', 'USD_CHF', 'GBP_USD']
    df = pd.read_csv(price_list)
    for major in majors:
        row = df[df['INSTRUMENT'] == major]
        high = row['HIGH'].values[0]
        low = row['LOW'].values[0]
        close = row['CLOSE'].values[0]
        if close < 0:
            sent = 'bearish'
        else:
            sent = 'bullish'
        twitter_text = 'Daily Forecast ' + major + ' High: ' + str(shorten(high)) + ' / Low: ' + str(shorten(low)) + ' Sentiment ' + sent + ' (' + str(round(100*abs(close),2)) + ' %)  - powered by themechnicalgerman.com #forex #' + major
        print(twitter_text)
        api.update_status(twitter_text)
