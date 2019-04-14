import dataService


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import backend as K
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.viewer import ImageViewer
from scipy.stats import gamma
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
import base64
import os

def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = imread(imgdata, plugin="imageio")
    return img

def get_prepared_data():

    data = dataService.getSavedData()
    sale = dataService.predictSale(data[0])
    print(dataService.predictSale(data[0]))

    dataset = []
    descriptions = []
    labels = []
    # folder = "adidas\\"
    # data_paths = os.listdir(folder)

    # Universal Sentance Encoder for the description.
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    # Import the Universal Sentence Encoder"s TF Hub module
    embed = hub.Module(module_url)

    # ResNet model for the images.
    base_model = ResNet50(weights="imagenet")

    for car in data:
        # Preparing the images.

        img = decode(car["image"])
        print(img.shape)
        img = resize(img, (224, 224), anti_aliasing=True) # Resnet feature extractor used 224x224 images, so doing this for simplicity.
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        image_features = base_model.predict(x)
        print(image_features.shape)

        # Preparing the metadata.
        fuel = car["Kuro tipas"]
        chasis = car["Kėbulo tipas"]
        city = car["Miestas"]
        wheel_position = car["Vairo padėtis"]
        date = car["Pagaminimo data"]
        cost = car["Kaina Lietuvoje"]
        mileage = car["Rida"]
        defects = car["Defektai"]
        # descriptions.append(description)
        prediction = car["Parduota per dienas (pagal taisykles)"]
        labels.append(prediction)

        joined_features = image_features[0] + fuel + chasis
        joined_features += price
        joined_features += size
        dataset.append(joined_features)

    # Description. Creating all description features in one bulk, because I don"t want to call session run for each product.
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        description_features = session.run(embed(descriptions))

    data = np.array(dataset)
    data = np.concatenate((data, description_features), axis=1)
    return data, labels

## TODO atkartoti su pilna f-ja
def get_gamma(elems):
  x = elems[0]
  shape = elems[1]
  N = 100
  a = tf.log(tf.reduce_mean(x) / shape)
  return tf.reduce_sum(tf.log(x)) * (shape - 1) - shape * N # - shape * N * tf.log(tf.reduce_mean(x) / shape)# - N * tf.lgamma(shape)


epsilon = 1.0e-9
def custom_objective(y_true, y_pred):
    # y_true - tikri laiko momentai, realizacijos
    # y_pred - modelio prognoze - parametrai
    elems = (y_true, y_pred)
    return K.map_fn(get_gamma, elems, dtype="float") 


data, labels = get_prepared_data()

droprate = 0.25

model = Sequential()
model.add(Dense(24, input_dim=1512, init="ones", activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(droprate))
model.add(Dense(8, init="ones", activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(droprate))
model.add(Dense(1, init="ones", activation="linear"))
model.compile(loss=custom_objective, optimizer="adam", metrics=["mse"])
model.fit(data, labels, epochs=1000, batch_size=32)