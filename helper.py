import os

import pandas as pd
import numpy as np
import cv2
import json
import errno

alpha = 0.0025

def preprocessing(img, steering=0.0, augmentation=True):
    new_steering = steering

    # crop 160x320 ==> 99x300
    img_tmp = img[40:139, 10:310, :]

    if augmentation:
        # shift
        # http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
        if np.random.randint(2) == 1:
            # the image width is 300, half is 150. I choose 1/3 of it.
            # Always shift to right, the next flip step may make it shift to left
            shift = np.random.randint(80)
            M = np.float32([[1,0,shift],[0,1,0]])
            img_tmp = cv2.warpAffine(img_tmp, M, (300, 99))
            new_steering += alpha * shift

        # flip
        if np.random.randint(2) == 1:
            img_tmp = img_tmp[:, ::-1, :]
            new_steering = -new_steering

        if new_steering > 1.0:
            new_steering = 1.0
        if new_steering < -1.0:
            new_steering = -1.0

    # resize 99x300 ==> 66x200
    img_tmp = cv2.resize(img_tmp, (200, 66))

    # normalize
    # img_tmp = img_tmp / 127.5 - 1.0

    return img_tmp, new_steering

def generate_arrays_from_dataframe(df, batch_size=32, augmentation=True):
    while True:
        for i in range(0, len(df), batch_size):
            X = []
            y = []
            for j in range(i, i+batch_size):
                if j >= len(df):
                    break
                filename = df.iloc[j]['images'].strip()
                steering = df.iloc[j]['steering']
                img = cv2.imread(os.path.join('data', filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                threshold = np.random.rand()
                while True:
                    img, steering = preprocessing(img, steering, augmentation)
                    if abs(steering) + 0.8 > threshold:
                        break
                X.append(img)
                y.append(steering)
            yield np.array(X), np.array(y)

def delete_file(filename):
    try:
        os.remove(filename)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise

def save_model(model, json_file='model.json', h5_file='model.h5'):
    delete_file(json_file)
    delete_file(h5_file)

    json_string = model.to_json()
    with open(json_file, 'w') as outfile:
        json.dump(json_string, outfile)
    model.save_weights(h5_file)
