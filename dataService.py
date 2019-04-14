import os
import codecs
import json
import base64
import pickle
from pprint import pprint

# KUO DAUGIAU TAŠKŲ TUO ILGIAU NEPARDUOS

def predictSale(car):
    soldInDays = 10
    try:
        soldInDays += pointsForDate(car['Pagaminimo data'])
        soldInDays += pointsForPrice(car['Kaina Lietuvoje'])
        soldInDays += pointsForMileage(car['Rida'])
        soldInDays += pointsForFuelType(car['Kuro tipas'])
        soldInDays += pointsForBodyType(car['Kėbulo tipas'])
        soldInDays += pointsForWheelPosition(car['Vairo padėtis'])
        soldInDays += pointsForDefects(car['Defektai'])
        soldInDays += pointsForLocation(car['Miestas'])
        
        car['Taškai už nuotrauką'] = pointsForImage(car['Defektai'], car['Pagaminimo data'], car['Spalva'], car['Kėbulo tipas'])
        soldInDays += pointsForImage(car['Defektai'], car['Pagaminimo data'], car['Spalva'],  car['Kėbulo tipas'])

        pass
    except Exception as e:
        print(e) 
        pass


    car['Parduota per dienas (pagal taisykles)'] = soldInDays

    return car

def pointsForDate(date):

    date = date.replace('-', '')
    if int(date) > 201500:
        return -5
    if int(date) > 201000:
        return 0
    if int(date) > 200500:
        return 5
    if int(date) > 200000:
        return 10
    return 15

def pointsForPrice(price):
    price = price.split('€', 1)[0].replace(' ', '')
    if int(price) > 15000:
        return 15
    if int(price) > 10000:
        return 10
    if int(price) > 8000:
        return 8
    if int(price) > 6000:
        return -3
    if int(price) > 2000:
        return -5
    if int(price) > 1000:
        return -8
    return -10

def pointsForMileage(mileage):
    if mileage == 'NA':
        return 5

    mileage = mileage.replace('km', '').replace(' ', '')
    if int(mileage) > 400000:
        return 10
    if int(mileage) > 300000:
        return 6
    if int(mileage) > 200000:
        return 4
    if int(mileage) > 100000:
        return -4
    return -6

def pointsForFuelType(fuel):
    if fuel == 'Dyzelinas':
        return -2
    if fuel == 'Benzinas':
        return 1
    return -3

def pointsForBodyType(bodyType):
    if bodyType == 'Hečbekas':
        return -2
    if bodyType == 'Sedanas':
        return -3
    if bodyType == 'Visureigis':
        return -1
    return 4

def pointsForWheelPosition(wheelPosition):
    if wheelPosition == 'Kairėje':
        return 0
    return 5
    
def pointsForDefects(defects):
    if defects == 'Be defektų':
        return 0
    return 4

def pointsForLocation(location):
    if location == 'Vilnius':
        return -3
    if location == 'Kaunas':
        return -2
    if location == 'Klaipėda':
        return -1
    if location == 'Marijampolė':
        return 0
    return 3

def pointsForImage(defected, date, color, bodyType):
    points = 0
    if defected == 'Be defektų':
        points += 0
    else:
        points += 5
    date = date.replace('-', '')

    if int(date) > 201500:
        points += 0
    elif int(date) > 201300:
        points += 3
    elif int(date) > 201100:
        points += 4
    elif int(date) > 200800:
        points += 5
    else:
        points += 6

    if color == 'Mėlyna / žydra':
        points += 5
    elif color == 'Raudona / vyšninė':
        points += 6

    if bodyType == 'Hečbekas':
        points += 4 
    if bodyType == 'Sedanas':
        points += 2
    if bodyType == 'Visureigis':
        points += 1

    return points


def labelDictionaryMaker(labelee, dictionary):
    if labelee not in dictionary:
        dictionary[labelee] = len(dictionary)  
        
    return dictionary

def exportImage(fileName, base64String):
    with open(fileName, 'wb') as f:
        f.write(base64.b64decode(base64String))


def getData():
    # Loading the data.
    with open('C:\\Users\\Ignas\\Documents\\Kursinis\\Evaluator\\data.json') as f:
        data = json.load(f)


    # Predicting sale time.
    for car in data:
        car = predictSale(car)

    # Exporting images to folders and assigning a rating
    # 1-st category is for the best cars
    dicImg = {}
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0

    for car in data:
        dicImg = labelDictionaryMaker(car['Taškai už nuotrauką'], dicImg)
        if car['Taškai už nuotrauką'] <= 5:
            car['Nuotraukos kategorija'] = 1
            #exportImage('C:\\Users\\Ignas\\Documents\\Kursinis\\CategorisedImages\\1st\\' + str(count1) + '.jpg', car['image'])
            count1 += 1
        elif car['Taškai už nuotrauką'] <= 6:
            car['Nuotraukos kategorija'] = 2
            #exportImage('C:\\Users\\Ignas\\Documents\\Kursinis\\CategorisedImages\\2nd\\' + str(count2) + '.jpg', car['image'])
            count2 += 1
        elif car['Taškai už nuotrauką'] <= 8:
            car['Nuotraukos kategorija'] = 3
            #exportImage('C:\\Users\\Ignas\\Documents\\Kursinis\\CategorisedImages\\3rd\\' + str(count3) + '.jpg', car['image'])
            count3 += 1
        elif car['Taškai už nuotrauką'] <= 11:
            car['Nuotraukos kategorija'] = 4
            #exportImage('C:\\Users\\Ignas\\Documents\\Kursinis\\CategorisedImages\\4th\\' + str(count4) + '.jpg', car['image'])
            count4 += 1
        else:
            car['Nuotraukos kategorija'] = 5
            #exportImage('C:\\Users\\Ignas\\Documents\\Kursinis\\CategorisedImages\\5th\\' + str(count5) + '.jpg', car['image'])
            count5 += 1

    # Processing.
    dicFuel = {}
    dicBody = {}
    dicFuel = {}
    dicWheel = {}
    dicLocation = {}

    for car in data:
        dicFuel = labelDictionaryMaker(car['Kuro tipas'], dicFuel)
        dicBody = labelDictionaryMaker(car['Kėbulo tipas'], dicBody)
        dicLocation = labelDictionaryMaker(car['Miestas'], dicLocation)
        dicWheel = labelDictionaryMaker(car['Vairo padėtis'], dicWheel)

    for car in data:
        car['Kuro tipas'] = dicFuel[car['Kuro tipas']]
        car['Kėbulo tipas'] = dicBody[car['Kėbulo tipas']]
        car['Miestas'] = dicLocation[car['Miestas']]
        car['Vairo padėtis'] = dicWheel[car['Vairo padėtis']]
        car['Pagaminimo data'] = car['Pagaminimo data'].replace('-', '')
        car['Kaina Lietuvoje'] = car['Kaina Lietuvoje'].split('€', 1)[0].replace(' ', '')
        car['Rida'] = car['Rida'].replace('km', '').replace(' ', '')
        if car['Rida'] == 'NA':
            car['Rida'] = 0

        # Special case.
        if car['Defektai'] == 'Be defektų':
            car['Defektai'] = 0
        else:
            car['Defektai'] = 1
 
    # Dumping data for later.
    with open('car.dictionary', 'wb') as config_dictionary_file:
        pickle.dump(data, config_dictionary_file)

    return data

def getSavedData():
    # Step 2
    with open('car.dictionary', 'rb') as config_dictionary_file:
    
        # Step 3
        data = pickle.load(config_dictionary_file)
    return data