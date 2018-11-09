######
# This is a copy from https://gist.github.com/pohzipohzi
# all credit goes to Poh Zi How


from bs4 import BeautifulSoup
import requests
import datetime
import logging
import csv
import dataset
import sys
import configparser


def set_logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='logs_file',
                        filemode='w')
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def get_myfxbook(table, config):
    # write to console current status
    logging.info("Connecting to myfxbook.com API")

    email = config.get('myfxbook', 'email')
    password = config.get('myfxbook', 'pwd')
    # login
    login_url = 'https://www.myfxbook.com/api/login.json?email=' + email + '&password=' + password
    response = requests.get(url=login_url)
    session = response.json().get('session')
    logging.info('Opened session {0}'.format(str(session)))
    # get the community position book
    com_url = 'http://www.myfxbook.com/api/get-community-outlook.json?session=' + session
    response = requests.get(url=com_url)
    # code.interact(banner='', local=locals())
    try:
        position_book = response.json()
    except Exception as e:
        print('Failed to get community outlook ' + str(e))
        return None
    myfxbook = position_book.get('symbols')
    logout_url = 'https://www.myfxbook.com/api/logout.json?session=' + session
    # log out
    requests.get(url=logout_url)
    date = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    for sym in myfxbook:
        database_entry = {'timestamp': date, **sym}
        table.upsert(database_entry, ['timestamp', 'name'])
        print(database_entry)
    logging.info("Successfully retrieved data")
    return None


if __name__ == "__main__":
    confname = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(confname)
    db = dataset.connect(config.get('data', 'myfxbook_path'))

    table = db['positionbook']

    get_myfxbook(table, config)