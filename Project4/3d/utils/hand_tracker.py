import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=1):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

    def get_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        landmarks = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = frame.shape
                    x, y, z = int(lm.x * w), int(lm.y * h), lm.z
                    landmarks.append((x, y, z))
        return landmarks
