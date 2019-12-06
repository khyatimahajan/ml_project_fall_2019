from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import glob
from skimage import io

dirImage = 'chest_xray_pneumonia/selected_dataset/'

filenames_train = []
filenames_train += glob.glob(dirImage+"training/*.jpg")

train_images = []
for file in filenames_train:
    img = np.asarray(cv.imread(file, 0))
    train_images.append(img)

train_images = np.asarray(train_images)

train_labels = []
with open(dirImage+'training/training_labels.txt', 'r') as rd:
    for line in rd:
        train_labels.append(int(line))
train_labels = np.asarray(train_labels, dtype=np.uint8)
# print('Training images: ', train_images.shape)
# print('Training labels: ', len(train_labels))

dirImage = 'chest_xray_pneumonia/selected_dataset/'

filenames_val = []
filenames_val += glob.glob(dirImage+"validation/*.jpg")

val_images = []
for file in filenames_val:
    img = np.asarray(cv.imread(file, 0))
    val_images.append(img)

val_images = np.asarray(val_images)

val_labels = []
with open(dirImage+'validation/validation_labels.txt', 'r') as rd:
    for line in rd:
        val_labels.append(int(line))
val_labels = np.asarray(val_labels, dtype=np.uint8)
# print('Validation images: ', val_images.shape)
# print('Validation labels: ', len(val_labels))

dirImage = 'chest_xray_pneumonia/selected_dataset/'

filenames_test = []
filenames_test += glob.glob(dirImage+"testing/*.jpg")

test_images = []
for file in filenames_test:
    img = np.asarray(cv.imread(file, 0))
    test_images.append(img)

test_images = np.asarray(test_images)

test_labels = []
with open(dirImage+'testing/testing_labels.txt', 'r') as rd:
    for line in rd:
        test_labels.append(int(line))
test_labels = np.asarray(test_labels, dtype=np.uint8)
# print('Testing images: ', test_images.shape)
# print('Testing labels: ', len(test_labels))


# def point_operation(image, threshold):
#     for row in range(0, image.shape[0]):
#         for col in range(0, image.shape[1]):
#             if image[row][col] >= threshold:
#                image[row][col] = 255
#             else:
#                image[row][col] = 0
#     return image

# def global_image_thresholding(
#     image=None, threshold=127, maxval=255, ttype=cv.THRESH_BINARY
# ):
#     ret, uimage = cv.threshold(image,threshold,maxval,ttype)
#     return uimage

# pt_train_images = []
# for i in train_images:
#     uimage = global_image_thresholding(i.copy(), threshold=160)
#     pt_train_images.append(uimage)
# pt_train_images = np.asarray(pt_train_images)

# pt_val_images = []
# for i in val_images:
#     uimage = global_image_thresholding(i.copy(), threshold=160)
#     pt_val_images.append(uimage)
# pt_val_images = np.asarray(pt_val_images)

# pt_test_images = []
# for i in test_images:
#     uimage = global_image_thresholding(i.copy(), threshold=160)
#     pt_test_images.append(uimage)
# pt_test_images = np.asarray(pt_test_images)

def median_filter(
    image=None, figure_size=9
):
    uimage = cv.medianBlur(image, figure_size)
    return uimage

mm_train_images = []
for i in train_images:
    uimage = median_filter(i.copy(), figure_size=9)
    mm_train_images.append(uimage)
mm_train_images = np.asarray(mm_train_images)

mm_val_images = []
for i in val_images:
    uimage = median_filter(i.copy(), figure_size=9)
    mm_val_images.append(uimage)
mm_val_images = np.asarray(mm_val_images)

mm_test_images = []
for i in test_images:
    uimage = median_filter(i.copy(), figure_size=9)
    mm_test_images.append(uimage)
mm_test_images = np.asarray(mm_test_images)

mm_train_images = mm_train_images / 255.0
mm_val_images = mm_val_images / 255.0
mm_test_images = mm_test_images / 255.0

preferred_height = 688 # need to change with the height dimension of each selected images
preferred_width = 1024 # need to change with the width dimension of each selected images
n_d=1
n_c=2
n_epoch=10 # number of epochs, we might wanna start with smaller values 

class Logger(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []

    def on_epoch_end(self, batch, logs={}):
        self.accuracy.append(logs.get('acc'))
logger = Logger()

model = keras.Sequential([
    keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(preferred_height, preferred_width, n_d)), 
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(64, (5,5), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(128, (5,5), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    keras.layers.Dense(n_c, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

mm_train_images = mm_train_images.reshape(mm_train_images.shape[0], preferred_height, preferred_width, n_d)
mm_val_images = mm_val_images.reshape(mm_val_images.shape[0], preferred_height, preferred_width, n_d)
mm_test_images = mm_test_images.reshape(mm_test_images.shape[0], preferred_height, preferred_width, n_d)

def shuffler(x, y):
    r_list = np.random.permutation(np.arange(x.shape[0]))
    return x[r_list], y[r_list]


mm_train_images, train_labels = shuffler(mm_train_images, train_labels)
model.fit(mm_train_images, train_labels, epochs=n_epoch, batch_size = 4,
          validation_data=(mm_val_images, val_labels), 
          callbacks=[logger])

result = model.evaluate(mm_test_images, test_labels, verbose=0)
print('Overall accuracy: %s%s'%(result[1]*100, '%'))

result = model.evaluate(mm_test_images[0:229], test_labels[0:229], verbose=0)
print('Class 1 accuracy: %s%s'%(result[1]*100, '%'))
result = model.evaluate(mm_test_images[229:], test_labels[229:], verbose=0)
print('Class 2 accuracy: %s%s'%(result[1]*100, '%'))
