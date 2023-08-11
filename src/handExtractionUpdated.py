import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import cv2 as cv
import numpy as np
import joblib
import os

from utils.imageSharpening import histogram_equalization, image_sharpening


# Updated mediapipe solutions March 2023
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmakerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image: mp.Image, hand_landmarks: list) -> mp.Image:
    """
    Inputs:
        rgb_image: image or frame to be processed
        hand_landmarks: list of landmarks for a single hand
    Outputs:
        annotated_image: image with landmarks drawn
    """
    annotated_image = np.copy(rgb_image)

    # Draw the hand landmarks
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


def record_landmarks(hand_landmarks: list) -> list[float]:
    coords_to_narray = []

    for landmark_point in hand_landmarks:
        coords_to_narray.append([landmark_point.x, landmark_point.y, landmark_point.z])
    
    return coords_to_narray


class Hand:
    def __init__(self, index: int, category: str, landmark_frame_l: list) -> None:
        self.index = index
        self.category = category
        self.landmarks_by_frame = []

        self.landmarks_by_frame.append(landmark_frame_l)

    def __str__(self) -> str:
        output_str = f'Hand #{self.index}, Category: {self.category}, Landmarks: {self.landmarks_by_frame}'
        return output_str
    
    def __repr__(self) -> str:
        return f'{self.category} Hand #{self.index}'
    
    def get_landmarks(self):
        return self.landmarks_by_frame

    def get_category(self):
        return self.category
    
    def get_index(self):
        return self.index
    
    def add_frame_landmarks(self, landmark_frame_l: list):
        self.landmarks_by_frame.append(landmark_frame_l)
    

def extract_hands(path: str, visualize: bool=False) -> dict():
    """
    Extracts landmarkers from video frames. Can visualize and save frame-by-frame.
    Hands are comprised of 21 landmarkers, each with designated (x, y, z) coordinates.
    ---------------------------------------------------------------------------------------------
    Args: 
        path: Path of video capture
        visualize: Visualize the landmark points onto each frame and save the frames as images. 
                   Default set to True.
    Returns:
        total_hand_list: Dictionary containing Hand(object)
    """

    # Obtain capture and video frame dimensions
    cap = cv.VideoCapture(path)
    WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv.CAP_PROP_FPS))
    VIDEO_OUTPUT_PATH = '/Users/elliot/Documents/NTU 2023/PDAnalysis/extractions/video'       #TODO issues with output_path and what it means
    FRAMES_OUTPUT_PATH = '/Users/elliot/Documents/NTU 2023/PDAnalysis/extractions/landmark_visuals'

    # Define codec and create a VideoWriter Object
    if visualize:
        print('Visualizing . . .')
        print('----------------------------')
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        output = cv.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT)) #TODO issues with this

    # Create a hand landmarker instance with video mode and specified options
    options = HandLandmakerOptions(
        base_options=BaseOptions(
        model_asset_path='/Users/elliot/Documents/NTU 2023/PDAnalysis/mediapipe_models/hand_landmarker.task'),
        num_hands=1, 
        min_hand_detection_confidence=0.5,
        running_mode=VisionRunningMode.VIDEO
        )
    
    frame_num = 0
    total_hands_list = {}
    # Build list of Hands frame by frame 
    # Each Hand records its landmarks for the given frame
    print('Building and Extracting Landmarks')
    print('------------------------------------------')
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                print('Failed to read frame or end of video!')
                break
            
            # Deblurring of each frame
            #frame = image_sharpening(frame)
            
            # Convert the frame received from OpenCV to a MediaPipe Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # Detect landmarks 
            timestamp = int(cap.get(cv.CAP_PROP_POS_MSEC))
            hand_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)
            if not hand_landmarker_result:
                continue
            
            # Analyze all hands within the frame and create landmarker visualizations
            for h_idx, all_frame_hand_list in enumerate(hand_landmarker_result.hand_landmarks):
                accuracy_score = hand_landmarker_result.handedness[h_idx][0].score
                # Record hand label
                hand_label = str(hand_landmarker_result.handedness[h_idx][0].category_name)
                # Obtain the 21 landmarks
                hand_landmarks_a = record_landmarks(all_frame_hand_list)

                # Create hand objects  
                if hand_label in total_hands_list and accuracy_score >= 0.9:
                    current_hand = total_hands_list[hand_label]
                    current_hand.landmarks_by_frame.append(hand_landmarks_a)
                elif hand_label not in total_hands_list:
                    print(f'Creating new Hand labeled {hand_label}')        # TODO If doesn't exist, put in random array that could be inaccurate. Need to address. Could use imputation.
                    current_hand = Hand(h_idx, hand_label, hand_landmarks_a)
                    total_hands_list[hand_label] = current_hand
                
                # Draw landmarkers on frame and save to file
                if visualize and accuracy_score >= 0.9:
                    frame_name = f'frame_{frame_num}.jpg'
                    annotated_frame = draw_landmarks_on_image(frame, all_frame_hand_list)
                    cv.imwrite(os.path.join(FRAMES_OUTPUT_PATH, frame_name), annotated_frame)

            frame_num += 1
            if frame_num >= 80 * 59:
                break
    
    ##TODO Uncomment to save landmarks as a .txt file 
    #joblib.dump(total_hands_list, f"{OUTPUT_PATH}{os.path.basename(path)[:-4]}.txt")

    cap.release()
    if visualize:
        output.release()
    print('---------------------------')
    print(f'Dictionary output of hands: {total_hands_list}')
    print('---------------------------')
    return total_hands_list


def preprocess_landmarks(extraction_dict: dict) -> list[Hand]:
    """
    Args:
        extraction_dict: Dictionary recieved from extract_hands
    Output:
        processed_list: List[Hand(object)]
    
    """
    processed_list = []
    for hand_obj in extraction_dict.values():
        processed_list.append(hand_obj)
    return processed_list


if __name__ == '__main__':
    print('Beginning Script . . .')
    print('--------------------------------')
    path_to_video = '/Users/elliot/Documents/NTU 2023/PDAnalysis/20200702_9BL.mp4'
    hand_dict = extract_hands(path_to_video, visualize=False)
    print('--------------------------------')
    print('Done extracting landmarks!')
    








