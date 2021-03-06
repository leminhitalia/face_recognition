from imutils.video import VideoStream
from imutils import face_utils
from imutils.face_utils import FaceAligner
import imutils
import time
import cv2
import dlib
import sys
import numpy as np
import json

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# load json and create model
model_json_file = open('model.json', 'r')
loaded_model_json = model_json_file.read()
model_json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("[INFO] Loaded model from disk")

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(2.0)

user_data = []
user_data_file = 'user_data.json'
try:
    with open(user_data_file) as input_file:
        user_data = json.load(input_file)
        input_file.close()
except IOError:
    print("[ERROR] Json file not found")

print("[DEBUG] user_data = {}".format(user_data))
# loop over the frames from the video stream
while True:
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    height, width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayayscale frame
    rects = detector(gray_frame, 0)

    # loop over the face detections
    for rect in rects:
        face_aligned = fa.align(frame, gray_frame, rect)
        face_aligned = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2GRAY)
        face_aligned = np.array(face_aligned)
        face_aligned = face_aligned.astype('float32')
        face_aligned /= 255.0
        face_aligned = np.expand_dims([face_aligned], axis=4)

        y_predict = loaded_model.predict(face_aligned)
        print("[DEBUG] y_predict = {}".format(y_predict))
        print("[DEBUG] y_predict[0] = {}".format(y_predict[0]))
        for index, ratio in enumerate(y_predict[0]):
            print("[DEBUG] user_index = {}, , ratio = {}".format(index, ratio))
            user = user_data[index]
            user_index = user['index']
            user_name = user['name']
            if str(index) != str(user_index):
                user_name = "Unknown"
                print("[ERROR] Incorrect predicting...index = {},  user_index = {}".format(index, user_index))

            result = user_name + ': ' + str(int(ratio * 100)) + '%'
            print("[DEBUG] result = {}".format(result))
            cv2.putText(frame, result, (10, 15 * (index + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # draw rect around face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        # draw person name
        result = np.argmax(y_predict, axis=1)
        print("[DEBUG] result = {}".format(result))
        print("[DEBUG] result[0] = {}".format(result[0]))
        user_index = result[0]
        user_name = "Unknown"
        try:
            user = user_data[user_index]
            user_name = user['name']
        except IndexError:
            print("[ERROR] IndexError: user_index out of range, user_index = {}".format(user_index))

        cv2.putText(frame, user_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # show the frame
    cv2.imshow("Frame", frame)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
sys.exit()

