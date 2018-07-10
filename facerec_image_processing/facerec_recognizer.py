import cv2
import json

# Create Local Binary Patterns Histograms for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('train_data/train_data.yml')

# Load prebuilt model for Frontal Face
cascadePath = "classifiers/haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath)

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

# Read user data
user_data_file = 'user_data/user_data.json'
user_data = []
try:
    with open(user_data_file) as input_file:
        user_data = json.load(input_file)
        input_file.close()
except IOError:
    print("[ERROR] Json file not found")

print("[DEBUG] user_data = {} ".format(user_data))

# Loop
while True:
    # Read the video frame
    ret, im = cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    # For each face in faces
    for (x, y, w, h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

        # Recognize the face belongs to which ID
        face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        face_id = str(face_id)

        print("[DEBUG] face_id = {}, confidence =  {}".format(face_id, confidence))

        # Check the ID if exist
        is_found_user = False
        for user in user_data:
            user_id = str(user['id'])
            print("[DEBUG] user_id = {}, face_id = {}, face_id equals user_id = {}".format(user_id, face_id, face_id == user_id))

            if face_id == user_id:
                face_id = user['name']
                is_found_user = True
                break

        if not is_found_user:
            face_id = 'Unknown'

        face_id = face_id + " {0:.2f}%".format(round(100 - confidence, 2))

        # Put text describe who is in the picture
        cv2.rectangle(im, (x - 22, y - 80), (x + w + 22, y - 22), (0, 255, 0), -1)
        cv2.putText(im, face_id, (x - 15, y - 35), font, 1, (255, 255, 255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('Face Recognition Window - Q: Quit', im)

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
