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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import seaborn as sns
import pandas
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


    # # Inspecting the data.
    # dfObj = pandas.DataFrame(data) 
    # print(dfObj.isna().sum())
    # dfObj = dfObj.dropna()
    # dfObj = dfObj.sample(frac=0.999,random_state=0)
    # dfObj = dfObj.astype(str)
    # sns.pairplot(dfObj[["Parduota per dienas (pagal taisykles)", "Kaina Lietuvoje", "Rida"]], diag_kind="kde")
    # plt.show()

    dataset = []
    descriptions = []
    labels = []
    # folder = "adidas\\"
    # data_paths = os.listdir(folder)

    # Universal Sentance Encoder for the description.
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    # Import the Universal Sentence Encoder"s TF Hub module
    # embed = hub.Module(module_url)

    # ResNet model for the images.
    base_model = ResNet50(weights="imagenet")

    progress = 0
    for car in data:

        progress = progress + 1
        print(str(progress) + " / 2151")

        try:
            # Preparing the images.
            img = decode(car["image"])
            print(img.shape)
            img = resize(img, (224, 224, 3), anti_aliasing=True) # Resnet feature extractor used 224x224 images, so doing this for simplicity.
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

            joined_features = []
            joined_features.append(int(fuel))
            joined_features.append(int(chasis))
            joined_features.append(int(city))
            joined_features.append(int(wheel_position))
            joined_features.append(int(date))
            joined_features.append(int(cost))
            joined_features.append(int(mileage))
            joined_features.append(int(defects))
            joined_features.extend(image_features[0])

            dataset.append(joined_features)


        except Exception as e:
            print(e)
            pass

    # Description. Creating all description features in one bulk, because I don"t want to call session run for each product.
    # with tf.Session() as session:
    #     session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    #     description_features = session.run(embed(descriptions))

    data = np.array(dataset)
    # data = np.concatenate((data, description_features), axis=1)

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

def plot_history(history):
  hist = pandas.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


data, labels = get_prepared_data()

training_data = data[0:1800]
training_labels = labels[0:1800]

validation_data = data[1800:]
validation_labels = labels[1800:]


droprate = 0.25

model = Sequential()        
model.add(Dense(2024, input_dim=1008, init="uniform", activation="relu")) # Shape with description = 1512
model.add(BatchNormalization())
model.add(Dropout(droprate))
model.add(Dense(512, init="uniform", activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(droprate))
model.add(Dense(1, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mean_absolute_error', 'mean_squared_error'])
# model.compile(loss=custom_objective, optimizer="adam", metrics=["mse"])
history = model.fit(training_data, training_labels, epochs=1000)

plot_history(history)
plt.show()

scores = model.evaluate(validation_data, validation_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))

predictions = model.predict(validation_data).flatten()
print(predictions)

plt.scatter(validation_labels, predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

plt.show()