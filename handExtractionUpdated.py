import mediapipe as mp
import cv2 as cv
import numpy as np
import pandas as pd
import os

# Updated mediapipe solutions March 2023
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmakerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def extract_hands(path):

    # Obtain capture and video frame dimensions
    # OUTPUT_PATH to be completed
    cap = cv.VideoCapture(path)
    WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv.CAP_PROP_FPS))
    OUTPUT_PATH = ''

    # Define codec and create a VideoWriter Object
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    output = cv.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    # Create a hand landmarker instance with video mode and specified options
    # Model path to be completed
    options = HandLandmakerOptions(
        base_options=BaseOptions(model_asset_path='????'),
        num_hands=1, 
        min_hand_detection_confidence=0.5,
        running_mode=VisionRunningMode.VIDEO
        )
    
    # Detect and build list of landmarks frame-by-frame
    all_landmarks = {'landmarks': [], 'hand_labels': []}

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()

            if not success:
                break
        
            # Convert the frame received from OpenCV to a MediaPipe Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

            
    





    







