# Copyright @2021 Ruining Li. All Rights Reserved.

import cv2
import mediapipe as mp
import numpy as np
import math
from keras.models import load_model

class GestureClassifier:
    '''

    '''
    def __init__(self):
        self._model = load_model("gesture_classification_model.h5")
        self._mp_hands = mp.solutions.hands

    def classify_gesture_for_frame(self, frame):
        '''Return 0 if the gesture is classified as paper; 1 if the gesture is classified as scissors; 2 if the gesture is classified as rock'''
        with self._mp_hands.Hands(max_num_hands=1) as hands:
            results = hands.process(cv2.flip(frame, 1))
            if not results.multi_hand_landmarks:
                return -1

            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints_data = []
            for idx in range(21):
                keypoints_data.append(hand_landmarks.landmark[idx].x)
                keypoints_data.append(hand_landmarks.landmark[idx].y)
                keypoints_data.append(hand_landmarks.landmark[idx].z)
            model_input = np.array(keypoints_data).reshape((1,63))

        return np.argmax(self._model.predict(model_input)[0])
