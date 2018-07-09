# Import OpenCV2 for image processing
# Import os for file path
import cv2
import json
# Import numpy for matrix calculation
import numpy as np
# Import Python Image Library (PIL)
from PIL import Image

import os

dataset_folder_name = "face_images/"
trainer_folder_name = "trainer/"

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Using prebuilt frontal face training model, for face detection
detector = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")

user_data_file = 'user_data/user_data.json'
face_images_folder = 'face_images/'

# Create method to get the images and label data
def getImagesAndLabels():

    user_data = []
    try:
        with open(user_data_file) as input_file:
            user_data = json.load(input_file)
            input_file.close()
    except IOError:
        print("[ERROR] Json file not found")

    faceSamples = []  # Initialize empty face sample
    ids = []  # Initialize empty id
    for user in user_data:
        print("[DEBUG] user = " + str(user))
        user_id = user['id']
        dir_path = face_images_folder + str(user_id)
        print("[DEBUG] dir_path = " + str(dir_path))
        if os.path.exists(dir_path):
            for img in os.listdir(dir_path):
                # Get the image and convert it to grayscale
                PIL_img = Image.open(dir_path + '/' + img).convert('L')
                # PIL image to numpy array
                img_numpy = np.array(PIL_img, 'uint8')
                # Get the face from the training images
                faces = detector.detectMultiScale(img_numpy)
                # Loop for each face, append to their respective ID
                for (x, y, w, h) in faces:
                    # Add the image to face samples
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    # Add the ID to IDs
                    ids.append(user_id)
        else:
            print("[ERROR] " + dir_path + " isn't exists.")

    print("[DEBUG] faceSamples = " + str(faceSamples))
    print("[DEBUG] ids = " + str(ids))

    # Pass the face array and IDs array
    return faceSamples, ids


# Get the faces and IDs
faces, ids = getImagesAndLabels()

# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
recognizer.save(trainer_folder_name + 'trainer.yml')