from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
import code
opts = Options()
opts.set_headless()
assert opts.headless  # Operating in headless mode
browser = Firefox(executable_path = '/home/ubuntu/driver/geckodriver', options=opts)
browser.set_page_load_timeout(20)
news_url = 'https://www.myfxbook.com/streaming-forex-news'
print('Loading site')
browser.get(news_url)

#data-gtmid="news-load-more"
print('looking for the button')
load_button = browser.find_elements_by_class('fxs_btn fxs_btn_line fxs_btn_block')
code.interact(banner = '',local = locals())
#load_button.click()