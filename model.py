import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Flatten, Dropout
from keras.layers.convolutional import Convolution2D

import helper

# hyperparameters
OFFSET = 0.2
EPOCHS = 8

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

driving_log = pd.read_csv(os.path.join('data', 'driving_log.csv'))
train_samples, validation_samples = train_test_split(driving_log, test_size=0.2)


center_train = train_samples[['center', 'steering']]
center_train.columns = ['images', 'steering']
left_train = train_samples[['left', 'steering']]
left_train.columns = ['images', 'steering']
left_train['steering'] += OFFSET
right_train = train_samples[['right', 'steering']]
right_train.columns = ['images', 'steering']
right_train['steering'] -= OFFSET
t_samples = center_train.append(left_train).append(right_train)
t_samples = shuffle(t_samples)

v_samples = validation_samples[['center', 'steering']]
v_samples.columns = ['images', 'steering']
v_samples = shuffle(v_samples)

history = model.fit_generator(helper.generate_arrays_from_dataframe(t_samples),
                              len(t_samples), EPOCHS,
                              validation_data=helper.generate_arrays_from_dataframe(v_samples),
                              nb_val_samples=len(v_samples))

helper.save_model(model)
