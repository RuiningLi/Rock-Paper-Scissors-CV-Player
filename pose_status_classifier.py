# Copyright @2021 Ruining Li. All Rights Reserved.

from collections import deque
import math
import numpy as np

class PoseStatusClassifier:
    '''

    '''
    def __init__(self):
        self.player_pose_status_buffer = deque()

    def update_player_pose_status_buffer(self, status):
        self.player_pose_status_buffer.append(status)
        if len(self.player_pose_status_buffer) > 3:
            self.player_pose_status_buffer.popleft()

    def _player_right_arm_angle(self, right_shoulder_world, right_elbow_world, right_wrist_world):
        upper_arm = np.array([right_shoulder_world.x - right_elbow_world.x,
                              right_shoulder_world.y - right_elbow_world.y,
                              right_shoulder_world.z - right_elbow_world.z])
        upper_arm = upper_arm / np.linalg.norm(upper_arm)
        forearm = np.array([right_wrist_world.x - right_elbow_world.x,
                            right_wrist_world.y - right_elbow_world.y,
                            right_wrist_world.z - right_elbow_world.z])
        forearm = forearm / np.linalg.norm(forearm)
        arm_angle = np.arccos(np.clip(np.dot(upper_arm, forearm), -1.0, 1.0))
        return arm_angle

    def is_player_going_to_play(self, right_shoulder_world, right_elbow_world, right_wrist_world):
        RIGHT_ARM_ANGLE_THRESHOLD = math.pi / 2
        return self._player_right_arm_angle(right_shoulder_world, right_elbow_world, right_wrist_world) > RIGHT_ARM_ANGLE_THRESHOLD
