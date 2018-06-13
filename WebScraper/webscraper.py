from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import requests
import base64
import json
import codecs
import os
import time
import datetime
from torrequest import TorRequest
with TorRequest(proxy_port=9050, ctrl_port=9051, password=None) as tr:

    def get_as_base64(url):
        return base64.b64encode(tr.get(url).content)

    def download_car_data(car_url, car, f):
        time.sleep(1)
        try:
            uClient = tr.get(car_url)
            car_page_html = uClient.text
            uClient.close()
            car_page_soup = soup(car_page_html, "html.parser")
            car_params = car_page_soup.findAll("table", {"class":"announcement-parameters"})[1].findAll("tr")

            car_dl = {'Pagaminimo data':'NA',
                'Rida':'NA',
                'Variklis':'NA',
                'Kuro tipas':'NA',
                'Kėbulo tipas':'NA',
                'Durų skaičius':'NA',
                'Varantieji ratai':'NA',
                'Pavarų dėžė':'NA',
                'Klimato valdymas':'NA',
                'Spalva':'NA',
                'Defektai':'NA',
                'Vairo padėtis':'NA',
                'Tech. apžiūra iki':'NA',
                'Ratlankių skersmuo':'NA',
                'Sėdimų vietų skaičius':'NA',
                'Pirmosios registracijos šalis':'NA',
                'Kaina Lietuvoje':'NA',
                'Modelis':'NA',
                'Miestas':'NA',
                'image':'NA'
                }
            
            car_dl['image'] = get_as_base64(car.div.div.img["src"]).decode("utf-8") 
            car_dl['Kaina Lietuvoje'] = car.find("div", {"class":"announcement-pricing-info"}).text.strip() 
            # car_dl['Modelis'] = car.find("div", {"class":"announcement-title"}).text.split(',')[0].replace("Skubiai", "").strip()
            # car_dl['Pagaminimo data'] = car.find("div", {"class":"announcement-parameters"}).find("span", {"title":"Pagaminimo data"}).text
            # car_dl['Kuro tipas'] = car.find("div", {"class":"announcement-parameters"}).find("span", {"title":"Kuro tipas"}).text 
            # car_dl['Pavarų dėžė'] = car.find("div", {"class":"announcement-parameters"}).find("span", {"title":"Pavarų dėžė"}).text 
            # car_dl['Rida'] = car.find("div", {"class":"announcement-parameters"}).find("span", {"title":"Rida"}).text  
            # car_dl['Miestas'] = car.find("div", {"class":"announcement-parameters"}).find("span", {"title":"Miestas"}).text 

            for child in car_params:
                car_dl[child.th.text] = child.td.text
                
            json_file = json.dumps(car_dl)
            f.write(json_file + ",")

        except  Exception as err: 
            print(err)
            pass
        
    def download_page_data(url, f):
        
        # Openning connection grabbing the page
        tr.reset_identity()
        try:
            uClient = tr.get(url)
        except Exception as err:
            print(err.info())
            pass

        header = uClient.status_code

        if header != 200:
            tr.reset_identity()
            download_page_data(url, f)
            
        page_html = uClient.text
        uClient.close()

        page_soup = soup(page_html, "html.parser")
        page_cars = page_soup.findAll("a", {"class":"announcement-item"})

        for car in page_cars:
            download_car_data(car["href"], car, f)


    my_url = 'https://autoplius.lt/skelbimai/naudoti-automobiliai?page_nr='
    filename = "data.json"
    f = codecs.open(filename,"w", "utf-8")
    f.write("[")

    i = 0
    for x in range(1, 250):
        print(str(x) + ":250")
        download_page_data(my_url + str(x), f)
        i += 1
        if i == 20:
            i = 0    
            time.sleep(10 * 60)
        


    f.close()
    with codecs.open(filename, 'rb+', "utf-8") as filehandle:
        filehandle.seek(-1, os.SEEK_END)
        filehandle.truncate()

    f = codecs.open(filename,"a", "utf-8")
    f.write("]")
    f.close()




