import os
import keras
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# tf.test.gpu_device_name()  # run to make sure tensorflow is connected to
import pictureindex
from PIL import Image

from skimage import transform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#model = ('./data/Saved Model/model11.h5')


#s = load_model(model)

# print(s.summary())

# print(model)

model = keras.models.load_model('model11.h5')

print(model.summary())


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (180, 180, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


image = load('8-12.png')

x = model.predict(image)

print(x)

# pictureindex.domino_list
