# Importing necessary libraries
import cv2
import numpy as np
import dlib
from imutils import face_utils

# Initialize the camera
cam = cv2.VideoCapture(0)

# If the camera fails to run
if not cam.isOpened():
    print("Error: Could not turn on the camera.")
    exit()

# Initialize the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\aksra\OneDrive\Documents\minor project_face det\shape_predictor_68_face_landmarks.dat")

# Status variables to monitor sleep, drowsiness, and activity
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Function to compute the Euclidean distance between two points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# Function to check if the eye is blinked
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if the eye is blinked based on the ratio
    if ratio > 0.25:
        return 2
    elif 0.21 <= ratio <= 0.25:
        return 1
    else:
        return 0

# Start capturing video
while True:
    ret, frame = cam.read()

    # If the frame was not captured correctly, break the loop
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Copy the frame before processing, so that face_frame is always defined
    face_frame = frame.copy()

    # Loop over each detected face
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Draw a rectangle around the face
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detect facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Detect blinking for the left and right eye
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Determine the user's status based on blinking
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING"
                #color = (255, 0, 0)

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy/Inactive"
                #color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                #color = (0, 0, 0)

        # Display the status on the frame
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw facial landmarks
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Only show the face detection result if faces are detected
    if len(faces) > 0:
        cv2.imshow("Result of detector", face_frame)

    # Exit the loop if 'ESC' key is pressed
    key = cv2.waitKey(1)
    if key == 27:  # 10 is the ASCII code for the enter key
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()

