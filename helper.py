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

    # convert it to YUV
    img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2YUV)

    # normalize
    img_tmp = img_tmp / 128.0 - 1.0

    return img_tmp

def generate_arrays_from_file(log_csv, batch_size=32):
    driving_log = pd.read_csv(os.path.join('data', log_csv))
    num_of_samples = len(driving_log)
    while True:
        X = []
        y = []

        for i in range(batch_size):
            # select idx
            idx = np.random.randint(0, num_of_samples)
            # select camera, 0 center, 1 left, 2 right
            camera = np.random.randint(0, 3)

            filename = driving_log.iloc[idx][camera].strip()
            steering = driving_log.iloc[idx]['steering']
            if camera == 1:
                steering += 0.15
            elif camera == 2:
                steering -= 0.15
            img = cv2.imread(os.path.join('data', filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocessing(img)
            # select mirror
            if np.random.randint(0, 2) == 1:
                img = img[:, ::-1, :]
                steering = -steering

            X.append(img)
            y.append(steering)
        assert len(X) == batch_size
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
