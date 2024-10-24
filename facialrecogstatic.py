import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import cosine_similarity
import zipfile
import os
from scipy.spatial import distance

# Load the Dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\aksra\OneDrive\Documents\minor project_face det\shape_predictor_68_face_landmarks.dat")

# Load the Dlib face recognition model
face_recognition_model = dlib.face_recognition_model_v1(r"C:\Users\aksra\OneDrive\Documents\minor project_face det\dlib_face_recognition_resnet_model_v1.dat")

# Function to extract face embeddings using Dlib
def get_face_embedding(image):
    # Convert image to RGB (Dlib works with RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get face detections
    faces = detector(rgb_image)

    if len(faces) == 0:
        raise ValueError("No faces detected in the image")

    # Assume the first detected face is the one we are interested in
    face_chip = dlib.get_face_chip(rgb_image, predictor, size=150)

    # Get face embedding
    face_descriptor = face_recognition_model.compute_face_descriptor(face_chip)

    # Convert to PyTorch tensor
    embedding = torch.tensor(face_descriptor)
    return embedding

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# Define the landmark indices for the eyes (from the 68 facial landmarks)
left_eye_indices = [36, 37, 38, 39, 40, 41]
right_eye_indices = [42, 43, 44, 45, 46, 47]

# EAR threshold and frame threshold for drowsiness detection
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 20

# Initialize frame counters
counter = 0
drowsy = False

# Load and process the image
image = cv2.imread(r"C:\Users\aksra\OneDrive\Documents\minor project_face det\Drowsy_datset\test\DROWSY\2.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

for face in faces:
    # Get the coordinates of the face
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Landmark detection and face alignment
    landmarks = predictor(gray, face)
    
    # Get the left and right eye landmarks
    left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in left_eye_indices])
    right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in right_eye_indices])

    # Calculate EAR for both eyes
    left_EAR = calculate_EAR(left_eye)
    right_EAR = calculate_EAR(right_eye)

    # Average EAR
    avg_EAR = (left_EAR + right_EAR) / 2.0

    # Check if the person is drowsy
    if avg_EAR < EAR_THRESHOLD:
        counter += 1
        if counter >= CONSECUTIVE_FRAMES:
            drowsy = True
            cv2.putText(image, "DROWSY", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        counter = 0
        drowsy = False
        cv2.putText(image, "ACTIVE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw the landmarks on the eyes
    for (x, y) in left_eye:
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
    for (x, y) in right_eye:
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

    # Crop the face from the image
    face_crop = image[y:y+h, x:x+w]

    # Get face embeddings
    embedding = get_face_embedding(face_crop)

    # Load known face embedding (assumed to be precomputed and saved)
    known_face_embedding = torch.load('/content/known_face_embedding.pth')

    # Compare face embeddings
    threshold = 0.8  # Define an appropriate threshold
    similarity = cosine_similarity(embedding.unsqueeze(0), known_face_embedding.unsqueeze(0))

    if similarity > threshold:
        print("Face recognized")
    else:
        print("Face not recognized")

# Save the result image
cv2.imwrite('/content/result_image.jpg', image)

# Final output: determine if the person is drowsy
if drowsy:
    print("Drowsy state detected")
else:
    print("Person is active")
