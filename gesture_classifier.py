# Copyright @2021 Ruining Li. All Rights Reserved.

import cv2
import numpy as np
import math
from keras.models import load_model

class GestureClassifier:
    def __init__(self):
        self._model = load_model("gesture_classification_model.h5")

    def classify_gesture(self, hand_landmarks):
        '''Return 0 if the gesture is classified as paper; 1 if the gesture is classified as scissors; 2 if the gesture is classified as rock'''
        hand_landmark_coordinates = []
        for i in range(21):
            hand_landmark_coordinates.append(hand_landmarks.landmark[i].x)
            hand_landmark_coordinates.append(hand_landmarks.landmark[i].y)
            hand_landmark_coordinates.append(hand_landmarks.landmark[i].z)
        model_input = np.array(hand_landmark_coordinates).reshape((1, 63))
        return np.argmax(self._model.predict(model_input)[0])
