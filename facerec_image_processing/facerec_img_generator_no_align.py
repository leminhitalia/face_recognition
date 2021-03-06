from imutils import face_utils
from imutils.face_utils import FaceAligner
import imutils
import cv2
import dlib
import os
import json
import datetime
import hashlib

# save_folder_name = input("Enter your id: ")
user_name = input("Enter your name: ")
save_folder_name = int(hashlib.sha256(user_name.encode('utf-8')).hexdigest(), 16) % (10**8)

base_dir = "face_images/"
if save_folder_name:
    base_dir = "{}{}/".format(base_dir, save_folder_name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

print("[INFO] Loading facial landmark predictor...")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("classifiers/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

# Load prebuilt model for Frontal Face
cascade_path = "classifiers/haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
face_cascade = cv2.CascadeClassifier(cascade_path)

print("[INFO] Camera sensor warming up...")
cam = cv2.VideoCapture(0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 800 pixels, and convert it to
    # gray scale
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=800)
    # height, width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the gray scale frame
    faces = face_cascade.detectMultiScale(gray_frame, 1.2, 5)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        user_data = []
        user_data_file = 'user_data/user_data.json'
        entry = {
            "id": save_folder_name,
            "name": user_name
        }
        try:
            with open(user_data_file) as input_file:
                user_data = json.load(input_file)
                input_file.close()
        except IOError:
            print("[ERROR] Json file not found")

        is_exist_user = False
        for user in user_data:
            if str(save_folder_name) == str(user['id']):
                is_exist_user = True
                break

        print("[DEBUG] is_exist_user = {}".format(is_exist_user))
        if not is_exist_user:
            user_data.append(entry)

        user_data = sorted(user_data, key=lambda user_key: user_key['id'], reverse=False)
        print("[DEBUG] user_date with index = {}".format(user_data))

        with open(user_data_file, mode='w') as output_file:
            output_file.write(json.dumps(user_data, indent=4))
            output_file.close()

        break

    # if the `s` key was pressed save the first found face
    if key == ord('s'):
        if len(faces) > 0:
            first_face = []
            first_face.append(faces[0])
            for (x, y, w, h) in first_face:
                image_name = "{}{}.png".format(base_dir, str(datetime.datetime.now()).replace(" ", "_").replace(":", "."))
                print("[DEBUG] image_name = {}".format(image_name))
                # save image
                cv2.imwrite(image_name, gray_frame[y:y + h, x:x + w])
                # show image
                cv2.imshow(image_name, gray_frame[y:y + h, x:x + w])

    # loop over the face detections
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # show the frame
    cv2.imshow("Face Detection Window - S: Save/Capture, Q: Quit", frame)

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
