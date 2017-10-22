from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

Network = VGG19
model = Network(weights="imagenet")

image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess(image)

preds = model.predict(image)
fv = preds.reshape((1000,1))
np.savetxt('pred.txt',fv)
