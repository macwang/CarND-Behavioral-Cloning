import os

import pandas as pd
import numpy as np
import cv2
import json
import errno

def preprocessing(img):
    # crop 160x320 ==> 99x300
    img_tmp = img[41:139, 10:310, :]

    # resize 99x300 ==> 66x200
    img_tmp = cv2.resize(img_tmp, (200, 66))

    # normalize
    img_tmp = img_tmp / 127.5 - 1.0

    return img_tmp

def generate_arrays_from_dataframe(df, batch_size=32):
    while True:
        for flip in range(2):
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
                    img = preprocessing(img)
                    if flip:
                        img = img[:, ::-1, :]
                        steering = -steering
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
