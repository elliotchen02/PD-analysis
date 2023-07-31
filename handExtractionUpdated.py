import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import cv2 as cv
import numpy as np
import os
import joblib

# Updated mediapipe solutions March 2023
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmakerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image, hand_landmarks):
    """
    Inputs:
        rgb_image: image or frame to be processed
        hand_landmarks: list of landmarks for a single hand
    Outputs:
        annotated_image: image with landmarks drawn
    """

    annotated_image = np.copy(rgb_image)

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
    annotated_image,
    hand_landmarks_proto,
    solutions.hands.HAND_CONNECTIONS,
    solutions.drawing_styles.get_default_hand_landmarks_style(),
    solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image

def record_landmarks(hand_landmarks):
    """
    Input:
        hand_landmarks: list of landmarks for a given hand
    Output:
        numpy array of landmark coordinates
    """
    coords_to_narray = []

    for landmark_point in hand_landmarks:
        coords_to_narray.append([landmark_point.x, landmark_point.y, landmark_point.z])
    
    return np.array(coords_to_narray)


def extract_hands(path, visualize=False):
    """
    Input: 
        path: Path of video capture
        visualize: Visualize the landmark points onto each frame and save the frames as images. 
                   Default set to True.
    
    extract_hands saves a dictionary that stores landmarkers and hand ID labels. 
    Landmarkers are stored in a list where each index corresponds to a hand identified in the frame.
    Hands are comprised of 21 landmarkers, each with designated (x, y, z) coordinates.
    
    Landmarkers structure:
        ['landmarkers'] = [
            [ (hand_index, numpy_array) ]    
        ]   

    numpy_array structure (21x3):
        array = [
            [x,y,z]
               .
               .
               .
            [x,y,z]      
        ]
        
    """

    # Obtain capture and video frame dimensions
    # TODO OUTPUT_PATH
    cap = cv.VideoCapture(path)
    WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv.CAP_PROP_FPS))
    OUTPUT_PATH = '/Users/elliot/Documents/NTU 2023/frames'

    # Define codec and create a VideoWriter Object
    if visualize:
        print('Visualizing')
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        output = cv.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    # Create a hand landmarker instance with video mode and specified options
    # TODO Model path
    options = HandLandmakerOptions(
        base_options=BaseOptions(model_asset_path='/Users/elliot/Documents/NTU 2023/PDAnalysis/mediapipe_models/hand_landmarker (1).task'),
        num_hands=1, 
        min_hand_detection_confidence=0.5,
        running_mode=VisionRunningMode.VIDEO
        )
    
    # Detect and build list of landmarks frame-by-frame
    all_landmarks = {'landmarks': [], 'hand_labels': []}
    all_landmarks_list = all_landmarks['landmarks']
    frame_num = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                print('Failed to read frame or end of video')
                break

            # Convert the frame received from OpenCV to a MediaPipe Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # Detect landmarks 
            timestamp = int(cap.get(cv.CAP_PROP_POS_MSEC))
            hand_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)
            print(hand_landmarker_result)
            
            all_landmarks_list.append([])
            if not hand_landmarker_result:
                continue
            
            # Analyze all hands within the frame and create landmarker visualizations
            for h_idx, hand_list in enumerate(hand_landmarker_result.hand_landmarks):
                if hand_landmarker_result.handedness[h_idx][0].score >= 0.95:
                    # Record hand label
                    all_landmarks['hand_labels'].append(hand_landmarker_result.handedness[h_idx][0].category_name)

                    # Record all landmark locations for the frame from list of landmarks
                    all_landmarks_list[frame_num].append( (h_idx, record_landmarks(hand_list)) )

                    # Draw landmarkers on frame and save to file
                    if visualize:
                        annotated_frame = draw_landmarks_on_image(frame.numpy_view(), hand_list)
                        output.write(annotated_frame)
                
            frame_num += 1
            if frame_num >= 80 * 59:
                break
    
    ## Save landmarks as a .txt file
    # joblib.dump(out_dt, f"{out_video_root}{hand}_hand_{os.path.basename(path)[:-4]}.txt")

    cap.release()
    if visualize:
        output.release()

    print(all_landmarks)

    return all_landmarks


if __name__ == '__main__':
    print('Beginning Script . . .')
    path_to_video = '/Users/elliot/Documents/NTU 2023/PDAnalysis/20200702_9BL.mp4'
    extract_hands(path_to_video, visualize=False)
    







