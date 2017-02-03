import os

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Flatten, Dropout
from keras.layers.convolutional import Convolution2D

import helper

# model start
model = Sequential()
model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2), input_shape=(66, 200, 3)))
model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile('adam', 'mse')
# model end

history = model.fit_generator(helper.generate_arrays_from_file('driving_log.csv'), 20000, 8)

helper.save_model(model)
