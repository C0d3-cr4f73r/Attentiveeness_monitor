import csv
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
predictions = []
timestamps = []

# Set CSV file path (replace with your desired path)
csv_file_path = "attentiveness_data.csv"

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results,font_scale=2, font_thickness=2):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
# Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    prediction = predict_attentiveness(results)
    text_x, text_y = (10, 30)  # Adjust coordinates as needed
    text = f"{prediction}"
    color = (0, 255, 0) if prediction == "Attentive" else (0, 0, 255)
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, color, font_thickness)

    return image
def predict_attentiveness(results):
    """Predicts attentiveness based on head and eye positions."""

    # Access pose and face landmarks
    pose_landmarks = results.pose_landmarks
    face_landmarks = results.face_landmarks

    # Define thresholds for attentiveness (adjust as needed)
    head_pitch_threshold = 15  # Degrees
    eye_aspect_ratio_threshold = 0.25
    head_pitch_angle=0


    # Calculate head pitch angle
    if pose_landmarks:
        nose = pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        ears = [
            pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR],
            pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR],
        ]
        head_pitch_angle = calculate_pitch_angle(nose, ears)

    # Calculate eye aspect ratio (EAR)
    if face_landmarks:
        eye_aspect_ratio = calculate_eye_aspect_ratio(face_landmarks)
    else:
        # Handle case where face landmarks are not detected
        eye_aspect_ratio = None

    # Determine attentiveness (consider both head pose and eye aspect ratio)
    if (
        
        head_pitch_angle < head_pitch_threshold
        and eye_aspect_ratio is not None
        and eye_aspect_ratio > eye_aspect_ratio_threshold
    ):
        return "Attentive"
    else:
        return "Not attentive"
  

def calculate_pitch_angle(nose, ears):
    """
    Calculates the pitch angle (in degrees) of the head based on nose and ear landmarks.

    Args:
        nose: Landmark object representing the nose tip.
        ears: List of two Landmark objects representing left and right ears.

    Returns:
        The pitch angle of the head in degrees.
    """

    # Convert landmarks to NumPy arrays for easier calculations
    nose_arr = np.array([nose.x, nose.y, nose.z])
    left_ear_arr = np.array([ears[0].x, ears[0].y, ears[0].z])
    right_ear_arr = np.array([ears[1].x, ears[1].y, ears[1].z])

    # Calculate the vector representing the head tilt direction
    head_tilt_vector = left_ear_arr - nose_arr

    # Calculate the norm of the vector
    head_tilt_vector_norm = np.linalg.norm(head_tilt_vector)

    # Calculate the z-axis unit vector
    z_axis = np.array([0, 0, 1])

    # Calculate the dot product of the head tilt vector and z-axis
    dot_product = np.dot(head_tilt_vector, z_axis)

    # Calculate the angle between the head tilt vector and z-axis using acos
    pitch_angle = np.arccos(dot_product / head_tilt_vector_norm)

    # Convert radians to degrees and return the angle
    return np.rad2deg(pitch_angle)
def calculate_eye_aspect_ratio(face_landmarks):
    """
    Calculates the eye aspect ratio (EAR) for both eyes based on facial landmarks.

    Args:
        face_landmarks: LandmarkList object containing facial landmarks.

    Returns:
        The average eye aspect ratio (EAR) for both eyes.
    """

    # Define eye landmark indices based on MediaPipe convention
    left_eye_landmarks = [36, 37, 38, 39, 40, 41]
    right_eye_landmarks = [42, 43, 44, 45, 46, 47]

    # Extract landmarks for each eye
    left_eye = [face_landmarks.landmark[i] for i in left_eye_landmarks]
    right_eye = [face_landmarks.landmark[i] for i in right_eye_landmarks]

    # Calculate EAR for each eye
    left_ear = calculate_individual_ear(left_eye)
    right_ear = calculate_individual_ear(right_eye)

    # Return the average EAR for both eyes
    return (left_ear + right_ear) / 2

def calculate_individual_ear(eye_landmarks):
    """
    Calculates the eye aspect ratio (EAR) for a single eye.

    Args:
        eye_landmarks: List of Landmark objects representing eye landmarks.

    Returns:
        The eye aspect ratio (EAR) for the given eye.
    """

    # Extract relevant landmark points
    p1 = eye_landmarks[0]
    p2 = eye_landmarks[1]
    p3 = eye_landmarks[2]
    p4 = eye_landmarks[3]
    p5 = eye_landmarks[4]
    p6 = eye_landmarks[5]

    # Calculate distances
    vertical_distance = np.linalg.norm(np.array([p3.x, p3.y]) - np.array([p6.x, p6.y]))
    horizontal_distance = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p4.x, p4.y]))

    # Calculate EAR and handle division by zero
    if horizontal_distance == 0:
        return 0
    else:
        return vertical_distance / horizontal_distance