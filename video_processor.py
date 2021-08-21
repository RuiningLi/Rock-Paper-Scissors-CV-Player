# Copyright @2021 Ruining Li. All Rights Reserved.

import cv2
import mediapipe as mp
from pose_status_classifier import *
from gesture_classifier import *
import math
from PIL import Image
from keras.models import load_model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose_status_classifier = PoseStatusClassifier()
gesture_classifier = GestureClassifier()
cap = cv2.VideoCapture(0)

# Get images from "assets/".
rock_gesture = cv2.imread("assets/rock.png")
paper_gesture = cv2.imread("assets/paper.png")
scissors_gesture = cv2.imread("assets/scissors.png")
empty = cv2.imread("assets/empty.png")
no_player_found = cv2.putText(cv2.imread("assets/empty.png"), "No Player Found! \n Don't dare to challenge me?", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

VISIBILITY_THRESHOLD = 0.8
player_next_move = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.1, min_tracking_confidence=0.1) as hand:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            pose_results = pose.process(image)
            hand_results = hand.process(cv2.flip(image, 1))

            if pose_results.pose_world_landmarks:
                right_wrist_world = pose_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                right_elbow_world = pose_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_shoulder_world = pose_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
                if min(right_wrist_world.visibility, right_elbow_world.visibility, right_shoulder_world.visibility) < VISIBILITY_THRESHOLD:
                    # Player's right wrist, right elbow, or right shoulder is not clear visible.
                    cv2.imshow("Let\'s play Rock, Paper, Scissors!", no_player_found)
                else:
                    pose_status_classifier.update_player_pose_status_buffer(
                        pose_status_classifier.is_player_going_to_play(right_shoulder_world, right_elbow_world, right_wrist_world))
                    if len(pose_status_classifier.player_pose_status_buffer) >= 3 and all(pose_status_classifier.player_pose_status_buffer):
                        print("Entered")
                        if not hand_results.multi_hand_landmarks:
                            continue
                        if player_next_move is None:
                            player_next_move = gesture_classifier.classify_gesture(hand_results.multi_hand_landmarks[0])
                        if player_next_move == 0:
                            cv2.imshow("Let\'s play Rock, Paper, Scissors!", scissors_gesture)
                        elif player_next_move == 1:
                            cv2.imshow("Let\'s play Rock, Paper, Scissors!", rock_gesture)
                        else:
                            cv2.imshow("Let\'s play Rock, Paper, Scissors!", paper_gesture)
                    else:
                        print("Exited")
                        if player_next_move is not None:
                            player_next_move = None
                        cv2.imshow("Let\'s play Rock, Paper, Scissors!", empty)
            else:
                # No player found.
                cv2.imshow("Let\'s play Rock, Paper, Scissors!", no_player_found)

            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release()
