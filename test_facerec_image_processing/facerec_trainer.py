# Import OpenCV2 for image processing
# Import os for file path
import cv2, json

# Import numpy for matrix calculation
import numpy as np

# Import Python Image Library (PIL)
from PIL import Image

import os


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


dataset_folder_name = "dataset/"
trainer_folder_name = "trainer/"

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Using prebuilt frontal face training model, for face detection
detector = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")


# Create method to get the images and label data
def getImagesAndLabels(path):
    # Get all file path
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    # Initialize empty face sample
    faceSamples = []

    # Initialize empty id
    ids = []

    non_duplicate_ids = []

    user_ids = []

    # Loop all the file path
    for imagePath in imagePaths:

        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')

        # PIL image to numpy array
        img_numpy = np.array(PIL_img, 'uint8')

        # Get the image id
        image_name = os.path.split(imagePath)[-1].split(".")
        id = int(image_name[1])

        # Get the face from the training images
        faces = detector.detectMultiScale(img_numpy)

        # Loop for each face, append to their respective ID
        for (x, y, w, h) in faces:
            # Add the image to face samples
            faceSamples.append(img_numpy[y:y + h, x:x + w])

            # Add the ID to IDs
            ids.append(id)

        if id not in non_duplicate_ids:
            non_duplicate_ids.append(id)
            user_ids.append({
                "name": image_name[0],
                "id": id
            })

    # Pass the face array and IDs array
    return faceSamples, ids, user_ids


# Get the faces and IDs
faces, ids, user_ids = getImagesAndLabels(dataset_folder_name)

# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
assure_path_exists(trainer_folder_name)
recognizer.save(trainer_folder_name + 'trainer.yml')

# Save user data
user_data_folder = 'user_data/'
assure_path_exists(user_data_folder)
user_data_file = 'user_data.json'
with open(user_data_folder + user_data_file, 'w+') as outfile:
    json.dump(user_ids, outfile, indent=4)
