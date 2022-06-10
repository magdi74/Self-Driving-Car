import csv
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout,convolutional ,pooling
import random
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

lines = []
with open('F:/Deep Learning Project/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for row in lines:
    path = row[0]
    image = cv2.imread(path)
    images.append(image)
    measurement = float(row[3])
    measurements.append(measurement)

def resize_image(image):
    image = cv2.cvtColor(cv2.resize(image[80:140,:], (32,32)), cv2.COLOR_BGR2RGB)
    return image

def generator_images(data, batchSize = 32):
    while True:
        for i in range(0, len(data), int(batchSize/4)):     #Steps 8 every time
            X_batch = []
            y_batch = []
            details = data[i: i+int(batchSize/4)]           #Takes all the stepped lines and processes them
            for line in details:
                image = resize_image(cv2.imread('./data/IMG/' + line[0].split('\\')[-1]))
                steering_angle = float(line[3])
                #appending original image
                X_batch.append(image)
                y_batch.append(steering_angle)
                #appending flipped image
                X_batch.append(np.fliplr(image))
                y_batch.append(-steering_angle)
                # appending left camera image and steering angle with offset
                X_batch.append(resize_image(cv2.imread('./data/IMG/' + line[1].split('\\')[-1])))
                y_batch.append(steering_angle+0.3)
                # appending right camera image and steering angle with offset
                X_batch.append(resize_image(cv2.imread('./data/IMG/' + line[2].split('\\')[-1])))
                y_batch.append(steering_angle-0.3)
            # converting to numpy array
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch

#X_train = np.array(images)
#y_train = np.array(measurements)
training_data, validation_data = train_test_split(lines, test_size = 0.2)

model = Sequential()
model.add(Lambda(lambda x: x /255.0 , input_shape=(32,32,3) ))
model.add(convolutional.Conv2D(15, 3, strides=(2, 2), activation = 'relu'))
model.add(Dropout(0.4))
model.add(pooling.MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(generator_images(training_data), steps_per_epoch = len(training_data)*4, epochs = 2, validation_data=generator_images(validation_data), validation_steps=len(validation_data))
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model2.h5')
