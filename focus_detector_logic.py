import cv2
import mediapipe as mp
import numpy as np

class FocusDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5)
        self.LEFT_EYE = [33, 133]
        self.RIGHT_EYE = [362, 263]

    def get_eye_aspect_ratio(self, landmarks, eye_indices, image_w, image_h):
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        x1, y1 = int(p1.x * image_w), int(p1.y * image_h)
        x2, y2 = int(p2.x * image_w), int(p2.y * image_h)
        dist = np.linalg.norm([x2 - x1, y2 - y1])
        return dist

    def is_focused(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = self.get_eye_aspect_ratio(landmarks, self.LEFT_EYE, w, h)
            right_ear = self.get_eye_aspect_ratio(landmarks, self.RIGHT_EYE, w, h)

            # if eye is almost closed or no face movement
            if left_ear < 10 or right_ear < 10:
                return False  # distracted
            else:
                return True   # focused
        else:
            return False  # no face detected → distracted
