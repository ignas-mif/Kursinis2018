import os
import codecs
import json
import base64
from pprint import pprint

def exportImage(fileName, base64String):
    with open(fileName, 'wb') as f:
        f.write(base64.b64decode(base64String))


# Loading the data.
with open('../Evaluator/data.json') as f:
    data = json.load(f)

for car in data:
    image = car['image']
    
    exportImage("1.jpg", image)