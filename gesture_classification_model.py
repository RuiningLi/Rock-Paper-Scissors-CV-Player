# Copyright @2021 Ruining Li. All Rights Reserved.

import cv2
import glob
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import mediapipe as mp
mp_hands = mp.solutions.hands

def get_keypoints_data(file_name):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        image = cv2.flip(cv2.imread(file_name), 1) 
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
    
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints_data = []
        for idx in range(21):
            keypoints_data.append(hand_landmarks.landmark[idx].x)
            keypoints_data.append(hand_landmarks.landmark[idx].y)
            keypoints_data.append(hand_landmarks.landmark[idx].z)
        return np.array(keypoints_data)

# Preprocess gesture images to get hand keypoints data
# 0 is the paper class; 1 is the scissors class; 2 is the rock class
paper_gesture_images = glob.glob('dataset/paper/*.png')
scissors_gesture_images = glob.glob('dataset/scissors/*.png')
rock_gesture_images = glob.glob('dataset/rock/*.png')
n_samples = len(paper_gesture_images) + len(scissors_gesture_images) + len(rock_gesture_images)
X = np.zeros((n_samples, 21*3))
y = np.zeros(n_samples)
    
cur_idx = 0
for file in paper_gesture_images:
    keypoints_data = get_keypoints_data(file)
    if keypoints_data is not None:
        X[cur_idx] = keypoints_data
        y[cur_idx] = 0
        cur_idx += 1
print("Paper:", str(cur_idx), "valid images in the dataset.")

for file in scissors_gesture_images:
    keypoints_data = get_keypoints_data(file)
    if keypoints_data is not None:
        X[cur_idx] = keypoints_data
        y[cur_idx] = 1
        cur_idx += 1
print("Scissors:", str(cur_idx), "valid images in the dataset.")

for file in rock_gesture_images:
    keypoints_data = get_keypoints_data(file)
    if keypoints_data is not None:
        X[cur_idx] = keypoints_data
        y[cur_idx] = 2
        cur_idx += 1
print("Rock:", str(cur_idx), "valid images in the dataset.")

X = X[:cur_idx, :]
y = y[:cur_idx]
one_hot_encoder = OneHotEncoder()
y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).toarray()
n_samples = cur_idx
print("There are " + str(n_samples) + " valid images in the dataset.")

# Train a neural network to classify hand keypoints data into its type
print("Training started.")
import keras
from keras.models import Sequential
from keras.layers import Dense
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = Sequential()
model.add(Dense(16, input_dim=63, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))

model.save("gesture_classification_model.h5")
