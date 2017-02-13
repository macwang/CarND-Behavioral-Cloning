
# Project 3 - CarND-Behavioral-Cloning

In this project, a video game-like car simulator is provided. Which records images from 3 front cameras(left, center and right) and the steering angles simultaneously. So assuming we manually drive it for a while, the recorded data can be used to train a DNN model that can simulate human's behavior.

## Dataset

Initially I wanted to generate my own dataset. However my 2014 MBP seems can't play the simulator smoothly so the car constantly drive to leave the roads even with human driver(in this case, me!). Many classmates suggested that the course provided data is enough to train a working model so I decided to stick on using that.

There are 8036 items in the driving_log.csv. It has 7 columns.
1. Center image filename (String)
2. Left image filename (String)
3. Right image filename (String)
4. Steering (Float)
5. Throttle (Float, not used in my model)
6. Brake (Float, not used in my model)
7. Speed (Float, not used in my model)

There are 4361 images with steering 0.0. It's too many then other steerings and might introduce bias. So I dropped 85% of it randomly. Then I split it to training set(80%) and validation set(20%). I only use center images for validation while using left/right images with compensate steerings for training.

## CNN Model

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        convolution2d_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       convolution2d_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1152)          0           flatten_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
____________________________________________________________________________________________________

##
