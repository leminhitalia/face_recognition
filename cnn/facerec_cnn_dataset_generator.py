from imutils.video import VideoStream
from imutils import face_utils
from imutils.face_utils import FaceAligner
import imutils
import time
import cv2
import dlib
import os
import json

user_data_file = 'user_data.json'
save_folder_name = int(input("Enter your id: "))
user_name = input("Enter your name: ")

base_dir = "face_images/"
if save_folder_name:
    base_dir = base_dir + str(save_folder_name) + "/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

print("[INFO] Camera sensor warming up...")
vs = VideoStream().start()
time.sleep(2.0)
count_image = 0

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 600 pixels, and convert it to
    # gray scale
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    height, width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the gray scale frame
    rects = detector(gray_frame, 0)


    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # if the `s` key was pressed save the first found face
    if key == ord('s'):
        if len(rects) > 0:
            faceAligned = fa.align(frame, gray_frame, rects[0])
            image_name = base_dir + str(count_image) + ".png"
            # save image
            cv2.imwrite(image_name, faceAligned)
            # show image
            cv2.imshow(image_name, faceAligned)
            count_image += 1
            if count_image > 20:
                count_image = 0

                user_data = []
                entry = {
                    "id": save_folder_name,
                    "name": user_name
                }
                try:
                    with open(user_data_file) as json_data:
                        user_data = json.load(json_data)
                        print("[INFO] user_data1: " + str(user_data))
                except IOError:
                    print("[WARN] Json file not found")

                user_data.append(entry)
                print("[INFO] user_data2: " + str(user_data))
                with open(user_data_file, mode='w') as f:
                    f.write(json.dumps(user_data, indent=4))

    # loop over the face detections
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

    # show the frame
    cv2.imshow("Frame", frame)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()