import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

news_url = 'https://www.fxstreet.com/news'

chrome_options = Options()
chrome_options.add_argument("--headless")
#chrome_options.binary_location = '/Applications/Google Chrome   Canary.app/Contents/MacOS/Google Chrome Canary'`

driver = webdriver.Chrome(chrome_options=chrome_options)
driver.get(news_url)
#data-gtmid="news-load-more"
load_button = driver.find_elements_by_tag_name('news-load-more')

load_button.click()
