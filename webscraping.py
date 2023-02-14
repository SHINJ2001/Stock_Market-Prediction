# import necessary library
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# open the Edge driver
driver = webdriver.Edge("C:\Program Files (x86)\msedgedriver.exe")

# set nasdaq url
nasdaq_url = 'https://www.nasdaq.com/market-activity/stocks/'

# set tickers list
ticker_dict = ['Apple' : 'AAPL', 'Amazon' : 'AMZN','Google': 'GOOGL', 
        'Microsoft' : 'MSFT', 'Tesla' : 'TSLA']

print('Enter the company whose stock prices you wish to predict')
print('Apple Amazon Google Microsoft Tesla')
select = input()
 
link = nasdaq_url + ticker_dict[select] + "/historical"

driver.get(link) # control the edge browser to navigate the corresponding 
driver.maximize_window() # maximize the window to prevent clicking wrong button

# Click "Accept all cookies" to proceed
wait = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//div[@id='onetrust-button-group']//button[@id='onetrust-accept-btn-handler']")))   
cookies_button = driver.find_element(By.XPATH, "//div[@id='onetrust-button-group']//button[@id='onetrust-accept-btn-handler']")
cookies_button.click()
    

# Click "MAX" button
wait = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//div[@class='table-tabs__list']//button[@class='table-tabs__tab' and text() = 'MAX']")))   
max_button = driver.find_element(By.XPATH, "//div[@class='table-tabs__list']//button[@class='table-tabs__tab' and text() = 'MAX']")
max_button.click()


# Click "Download Data" button
wait = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//div[@class='historical-data__controls']//button[@class='historical-data__controls-button--download historical-download']//*[name()='svg']")))   
download_button = driver.find_element(By.XPATH, "//div[@class='historical-data__controls']//button[@class='historical-data__controls-button--download historical-download']//*[name()='svg']")
download_button.click()

print('Finished downloading historical price data of ' + ticker)

driver.close() # close the Edge browser
time.sleep(15) # wait until the last excel file downloaded

# save the consolidated workbook
print('Finished consolidating the price data for all tickers')
