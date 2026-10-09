import cv2
import numpy as np
from collections import deque, Counter
import mediapipe as mp
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ISLPredictor:
    def __init__(self):
        self.model_path = "model.h5"
        self.sequence_length = 30
        self.features = 63
        self.actions = np.array(['you', 'good', 'thank you', 'me', 'my', 'hello'])
        
        # Load Model
        self.model = load_model(self.model_path)
        
        # Mediapipe Setup
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,  
            num_hands=2,  
            min_hand_detection_confidence=0.7,  
            min_hand_presence_confidence=0.7,  
            min_tracking_confidence=0.7
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        self.sequence = deque(maxlen=self.sequence_length)
        self.prediction_history = deque(maxlen=5)
        self.threshold = 0.70
        self.detected_words_list = []
        self.last_prediction = ""

    def extract_keypoints(self, result):
        if result.hand_landmarks:  
            hand = result.hand_landmarks[0]  
            base_x = hand[0].x  
            base_y = hand[0].y  
            base_z = hand[0].z  
            pts = []  
            for lm in hand:  
                pts.append(lm.x - base_x)  
                pts.append(lm.y - base_y)  
                pts.append(lm.z - base_z)  
            return np.array(pts, dtype=np.float32)  
        else:  
            return np.zeros(self.features, dtype=np.float32)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)

        hand_detected = bool(detection_result.hand_landmarks)
        current_prediction = ""
        confidence = 0.0

        if not hand_detected:
            self.sequence.clear()
        else:
            keypoints = self.extract_keypoints(detection_result)
            self.sequence.append(keypoints)

            if len(self.sequence) == self.sequence_length:
                input_data = np.expand_dims(np.array(self.sequence), axis=0)
                res = self.model.predict(input_data, verbose=0)[0]
                predicted_index = np.argmax(res)
                confidence = float(res[predicted_index])

                if confidence > self.threshold:
                    self.prediction_history.append(predicted_index)

                if len(self.prediction_history) > 0:
                    most_common = Counter(self.prediction_history).most_common(1)[0][0]
                    current_prediction = self.actions[most_common]

                if current_prediction and current_prediction != self.last_prediction:
                    self.detected_words_list.append(current_prediction)
                    self.last_prediction = current_prediction

        return {
            "hand_detected": hand_detected,
            "prediction": current_prediction if current_prediction else "...",
            "confidence": confidence,
            "collecting": len(self.sequence),
            "sequence_length": self.sequence_length,
            "words": self.detected_words_list
        }