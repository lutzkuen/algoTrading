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


def get_economic_calendar(startlink, sq_table):
    # write to console current status
    logging.info("Scraping data for link: {}".format(startlink))

    # get the page and make the soup
    base_url = "https://www.forexfactory.com/"
    r = requests.get(base_url + startlink)
    data = r.text
    soup = BeautifulSoup(data, "lxml")

    # get and parse table data, ignoring details and graph
    table = soup.find("table", class_="calendar__table")

    # do not use the ".calendar__row--grey" css selector (reserved for historical data)
    trs = table.select("tr.calendar__row.calendar_row")
    fields = ["date", "time", "currency", "impact", "event", "actual", "forecast", "previous"]

    # some rows do not have a date (cells merged)
    curr_year = startlink[-4:]
    curr_date = ""
    curr_time = ""
    for tr in trs:

        # fields may mess up sometimes, see Tue Sep 25 2:45AM French Consumer Spending
        # in that case we append to errors.csv the date time where the error is
        try:
            for field in fields:
                data = tr.select("td.calendar__cell.calendar__{}.{}".format(field, field))[0]
                # print(data)
                if field == "date" and data.text.strip() != "":
                    curr_date = data.text.strip()
                elif field == "time" and data.text.strip() != "":
                    # time is sometimes "All Day" or "Day X" (eg. WEF Annual Meetings)
                    if data.text.strip().find("Day") != -1:
                        curr_time = "12:00am"
                    else:
                        curr_time = data.text.strip()
                elif field == "currency":
                    currency = data.text.strip()
                elif field == "impact":
                    # when impact says "Non-Economic" on mouseover, the relevant
                    # class name is "Holiday", thus we do not use the classname
                    impact = data.find("span")["title"]
                elif field == "event":
                    event = data.text.strip()
                elif field == "actual":
                    actual = data.text.strip()
                elif field == "forecast":
                    forecast = data.text.strip()
                elif field == "previous":
                    previous = data.text.strip()

            dt = datetime.datetime.strptime(",".join([curr_year, curr_date, curr_time]),
                                            "%Y,%a%b %d,%I:%M%p")
            news_object = {'date': dt.strftime('%Y-%m-%d'), 'time': curr_time, 'currency': currency, 'impact': impact,
                    'event': event, 'actual': actual, 'forecast': forecast, 'previous': previous}
            sq_table.upsert(news_object, ['year', 'date', 'time', 'currency', 'event'])
        except:
            with open("errors.csv", "a") as f:
                csv.writer(f).writerow([curr_year, curr_date, curr_time])

    logging.info("Successfully retrieved data")
    return


if __name__ == "__main__":
    """
    Run this using the command "python `script_name`.py >> `output_name`.csv"
    """
    try:
        mode = sys.argv[1]
        confname = sys.argv[2]
        config = configparser.ConfigParser()
        config.read(confname)
        db = dataset.connect(config.get('data', 'calendar_path'))
    except Exception as e:
        print(e)
        mode = None
    table = db['calendar']
    set_logger()
    if mode == 'full_load':
        ts0 = datetime.datetime.strptime('2009-01-01', '%Y-%m-%d')  # lets go back in time
        ts1 = datetime.datetime.now()
        while ts0 < ts1:
            now = ts0  # datetime.datetime.now()
            now += datetime.timedelta(days=(6 - now.weekday()))
            now_mon = now.strftime('%b').lower()
            now_day = now.strftime('%d').lstrip('0')
            now_year = now.strftime('%Y')
            endlink = 'calendar.php?week=' + now_mon + now_day + '.' + now_year
            get_economic_calendar(endlink, table)
            ts0 = ts0 + datetime.timedelta(days=7)
    if mode == 'delta_load':
        now = datetime.datetime.now()
        now += datetime.timedelta(days=(6 - now.weekday()))
        now_mon = now.strftime('%b').lower()
        now_day = now.strftime('%d').lstrip('0')
        now_year = now.strftime('%Y')
        then = now - datetime.timedelta(days=7)
        then_mon = then.strftime('%b').lower()
        then_day = then.strftime('%d').lstrip('0')
        then_year = then.strftime('%Y')
        endlink = 'calendar.php?week=' + now_mon + now_day + '.' + now_year
        startlink = 'calendar.php?week=' + then_mon + then_day + '.' + then_year
        get_economic_calendar(startlink, table)
        get_economic_calendar(endlink, table)
