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