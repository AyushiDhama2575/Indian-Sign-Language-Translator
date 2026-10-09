import os
import base64
import json
from collections import Counter, deque
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

from google import genai
from google.genai import types

load_dotenv(find_dotenv())

app = FastAPI()

# 1. Static files mount for videos folder
LOCAL_FOLDER = r"C:\Users\SHREE RAM\prac_isl\videos"
if os.path.exists(LOCAL_FOLDER):
    app.mount("/videos", StaticFiles(directory=LOCAL_FOLDER), name="videos")

# 2. CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Gemini Client Initialization
client = genai.Client(api_key="Gemini API")

class TranslateRequest(BaseModel):
    text: str
    language: str

class SentenceRequest(BaseModel):
    words: list
    language: str

def get_gemini_completion(prompt: str, model="gemini-3.6-flash"):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
        ),
    )
    return response.text

# Local fast translation map for live webcam predictions
LOCAL_TRANSLATIONS = {
    "Hindi": {
        "you": "आप",
        "good": "अच्छा",
        "thank you": "धन्यवाद",
        "me": "मैं",
        "my": "मेरा",
        "hello": "नमस्ते"
    },
    "Telugu": {
        "you": "మీరు",
        "good": "మంచిది",
        "thank you": "ధన్యవాదాలు",
        "me": "నేను",
        "my": "నా",
        "hello": "నమస్కారం"
    },
    "Bengali": {
        "you": "আপনি",
        "good": "ভালো",
        "thank you": "ধন্যবাদ",
        "me": "আমি",
        "my": "আমার",
        "hello": "নমস্কার"
    }
}

def translate_word_locally(word: str, language: str) -> str:
    word_lower = word.lower().strip()
    if language in LOCAL_TRANSLATIONS and word_lower in LOCAL_TRANSLATIONS[language]:
        return LOCAL_TRANSLATIONS[language][word_lower]
    return word

# ---------------------------------------------------------
# ISL PREDICTOR CLASS (Webcam & Model Logic)
# ---------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  
    (0, 5), (5, 6), (6, 7), (7, 8),  
    (0, 9), (9, 10), (10, 11), (11, 12),  
    (0, 13), (13, 14), (14, 15), (15, 16),  
    (0, 17), (17, 18), (18, 19), (19, 20),  
    (5, 9), (9, 13), (13, 17)
]

class ISLPredictor:
    def __init__(self):
        self.model_path = "model.h5"
        self.sequence_length = 30
        self.features = 63
        self.actions = np.array(['you', 'good', 'thank you', 'me', 'my', 'hello'])
        
        self.model = load_model(self.model_path)
        
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

    def draw_landmarks(self, image, result):
        if not result.hand_landmarks:
            return image
        h, w, _ = image.shape
        for hand_landmarks in result.hand_landmarks:
            for start, end in HAND_CONNECTIONS:
                x1 = int(hand_landmarks[start].x * w)
                y1 = int(hand_landmarks[start].y * h)
                x2 = int(hand_landmarks[end].x * w)
                y2 = int(hand_landmarks[end].y * h)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
        return image

    def process_frame(self, frame, selected_language="English"):
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

        annotated_frame = self.draw_landmarks(frame, detection_result)
        localized_prediction = translate_word_locally(current_prediction, selected_language) if current_prediction else "..."

        return {
            "annotated_frame": annotated_frame,
            "hand_detected": hand_detected,
            "prediction": localized_prediction,
            "confidence": confidence,
            "collecting": len(self.sequence),
            "sequence_length": self.sequence_length,
            "words": self.detected_words_list
        }

    def generate_gemini_sentence(self, selected_language="English"):
        if len(self.detected_words_list) == 0:
            return "..."
        
        words_string = ", ".join(self.detected_words_list)
        prompt = f"""These are sign language detected words in sequence: [{words_string}]. 
Using your intelligence, form a single, natural, and meaningful sentence out of these words, 
and strictly output the final sentence translated into this language: {selected_language}. 
Give only the final sentence without extra explanation."""
        try:
            return get_gemini_completion(prompt, model="gemini-3.6-flash").strip()
        except Exception:
            return " ".join(self.detected_words_list)

# ---------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------

@app.post("/translate-isl")
def translate_to_isl(request: TranslateRequest):
    user_input = request.text
    selected_language = request.language
    
    try:
        if not os.path.exists(LOCAL_FOLDER):
            raise HTTPException(status_code=404, detail="Videos folder not found!")
            
        video_files = [f for f in os.listdir(LOCAL_FOLDER) if f.endswith('.mp4')]
        available_words = [os.path.splitext(f)[0] for f in video_files]
        
        if not available_words:
            raise HTTPException(status_code=404, detail="No video files found in the folder!")
            
        prompt = f"""
        You are an expert Indian Sign Language (ISL) grammar and translation engine.
        The user has provided an input sentence in '{selected_language}': "{user_input}"

        Here is the list of available sign language video words we have in our storage:
        {available_words}

        Task & Rules:
        1. Translate and break down the sentence structurally into Indian Sign Language flow.
        2. Strictly map/match each concept to the closest available words from the provided list. Do not invent words.
        3. CRITICAL REQUIREMENT: Output your response entirely in the user's chosen language ('{selected_language}').
        4. Return ONLY the matching words separated by commas in the correct sequence.
        """
        
        gemini_response = get_gemini_completion(prompt)
        matched_words = [word.strip().lower() for word in gemini_response.strip().split(",") if word.strip()]
        
        video_sequence = []
        for word in matched_words:
            file_name = f"{word}.mp4"
            full_path = os.path.join(LOCAL_FOLDER, file_name)
            
            if os.path.exists(full_path):
                video_sequence.append({
                    "word": word,
                    "video_file": file_name,
                    "stream_url": f"http://localhost:8000/videos/{file_name}"
                })
                
        return {
            "selected_language": selected_language,
            "original_text": user_input,
            "matched_sequence": video_sequence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Added POST route back for compatibility if frontend uses fetch instead of websocket for AI sentence
@app.post("/generate-sentence-ai")
def generate_sentence_ai(request: SentenceRequest):
    try:
        if not request.words:
            return {"sentence": "No words detected yet."}
            
        words_string = ", ".join(request.words)
        prompt = f"""These are sign language detected words in sequence: [{words_string}]. 
Using your intelligence, form a single, natural, and meaningful sentence out of these words, 
and strictly output the final sentence translated into this language: {request.language}. 
Give only the final sentence without extra explanation."""
        
        smart_sentence = get_gemini_completion(prompt, model="gemini-3.6-flash").strip()
        return {"sentence": smart_sentence}
    except Exception as e:
        return {"sentence": " ".join(request.words)}

@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    await websocket.accept()
    predictor = ISLPredictor()
    print("Webcam client connected.")
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            msg_type = data.get("type")
            selected_language = data.get("language", "English")

            if msg_type == "frame":
                image_data = data.get("image")
                if "," in image_data:
                    image_data = image_data.split(",", 1)[1]

                image_bytes = base64.b64decode(image_data)
                np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                result = predictor.process_frame(frame, selected_language)

                _, buffer = cv2.imencode('.jpg', result["annotated_frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                encoded_frame = base64.b64encode(buffer).decode('utf-8')

                await websocket.send_json({
                    "type": "prediction",
                    "annotated_image": f"data:image/jpeg;base64,{encoded_frame}",
                    "hand_detected": result["hand_detected"],
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "collecting": result["collecting"],
                    "sequence_length": result["sequence_length"],
                    "words": result["words"]
                })
            
            elif msg_type == "generate_sentence":
                sentence = predictor.generate_gemini_sentence(selected_language)
                await websocket.send_json({
                    "type": "gemini_sentence",
                    "sentence": sentence
                })

    except WebSocketDisconnect:
        print("Webcam client disconnected.")