import cv2
import os
import numpy as np
import json

from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# load and prepare data
user_data_file = 'user_data.json'
face_images_folder = 'face_images/'


# def load_data(user_data_file, face_images_folder):
user_data = []
user_data_file = user_data_file
try:
    with open(user_data_file) as input_file:
        user_data = json.load(input_file)
        input_file.close()
except IOError:
    print("[ERROR] Json file not found")

img_data_list = []
labels = []
valid_images = [".jpg", ".gif", ".png"]
for user in user_data:
    print("[DEBUG] user = " + str(user))
    user_id = user['id']
    dir_path = face_images_folder + str(user_id)
    for img in os.listdir(dir_path):
        name, ext = os.path.splitext(img)
        if ext.lower() not in valid_images:
            continue

        img_data = cv2.imread(dir_path + '/' + img)
        # convert image to gray
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        img_data_list.append(img_data)
        labels.append(str(user_id))

img_data_list = np.array(img_data_list)
img_data_list = img_data_list.astype('float32')
labels = np.array(labels, dtype='int64')
# scale down(so easy to work with)
img_data_list /= 255.0
img_data_list = np.expand_dims(img_data_list, axis=4)
print("[DEBUG] img_data_list.shape = " + str(img_data_list.shape))
print("[DEBUG] img_data_list.shape[0] = " + str(img_data_list.shape[0]))
print("[DEBUG] img_data_list.shape = " + str(img_data_list.shape))
print("[DEBUG] labels.shape = " + str(labels.shape))

# convert class labels to on-hot encoding
user_data_len = len(user_data)
y = np_utils.to_categorical(labels, 1000)
print("[INFO] y = " + str(y))

# Shuffle the dataset
x, y = shuffle(img_data_list, y, random_state=2)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print("[DEBUG] y_train.shape = " + str(y_train.shape))
print("[DEBUG] y_test.shape = " + str(y_test.shape))

# Defining the model
input_shape = img_data_list[0].shape
print("[DEBUG]" + str(input_shape))

#    return x_train, x_test, y_train, y_test, input_shape, user_data_len