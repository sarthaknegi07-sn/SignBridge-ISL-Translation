import os
import sys
import cv2
import torch
import numpy as np
import pickle
import mediapipe as mp
import time
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ISLModel
from utils.keypoint_utils import NUM_FRAMES, NUM_KEYPOINTS, normalize_sequence
from voice.speaker import speak, precache_phrases
from voice.voice_config import CONFIDENCE_THRESHOLD, COOLDOWN_SECONDS

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_live_keypoints(results):
    """Extract keypoints from MediaPipe live results"""
    keypoints = np.zeros(NUM_KEYPOINTS)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            offset = hand_idx * 63
            for i, lm in enumerate(hand_landmarks.landmark):
                if offset + i * 3 + 2 < NUM_KEYPOINTS:
                    keypoints[offset + i * 3] = lm.x
                    keypoints[offset + i * 3 + 1] = lm.y
                    keypoints[offset + i * 3 + 2] = lm.z

    return keypoints

def live_detect():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Load label encoder
    with open(f"{PROCESSED_DIR}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)
    phrases = list(le.classes_)

    # Load model
    model = ISLModel(
        input_size=126,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(torch.load(f"{MODELS_DIR}/best_model.pth", map_location=device))
    model.eval()
    print("Model loaded!")

    # Pre-cache audio
    precache_phrases(phrases)

    # Rolling frame buffer
    sequence_buffer = []
    last_spoken_time = 0
    last_spoken_phrase = ""
    current_phrase = ""
    current_confidence = 0.0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=2
    ) as hands:

        print("\nLive detection started! Press Q to quit.\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Extract keypoints
            keypoints = extract_live_keypoints(results)
            sequence_buffer.append(keypoints)

            # Keep only last NUM_FRAMES
            if len(sequence_buffer) > NUM_FRAMES:
                sequence_buffer.pop(0)

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            # Predict when buffer is full
            if len(sequence_buffer) == NUM_FRAMES:
                sequence = np.array(sequence_buffer, dtype=np.float32)
                sequence = normalize_sequence(sequence)
                tensor = torch.tensor(sequence).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, pred_idx = torch.max(probs, 1)
                    confidence = confidence.item()
                    pred_idx = pred_idx.item()

                if confidence >= CONFIDENCE_THRESHOLD:
                    current_phrase = le.classes_[pred_idx]
                    current_confidence = confidence

                    # Speak if cooldown passed and phrase changed
                    now = time.time()
                    if (now - last_spoken_time > COOLDOWN_SECONDS and
                            current_phrase != last_spoken_phrase):
                        threading.Thread(
                            target=speak,
                            args=(current_phrase,),
                            daemon=True
                        ).start()
                        last_spoken_time = now
                        last_spoken_phrase = current_phrase

            # UI overlay
            h, w = frame.shape[:2]

            # Background box
            cv2.rectangle(frame, (0, h - 120), (w, h), (0, 0, 0), -1)

            # Phrase display
            if current_phrase:
                cv2.putText(frame, current_phrase.upper(),
                    (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {current_confidence*100:.1f}%",
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Buffer progress bar
            buffer_pct = len(sequence_buffer) / NUM_FRAMES
            bar_w = int(w * buffer_pct)
            cv2.rectangle(frame, (0, h - 125), (bar_w, h - 120), (0, 255, 0), -1)

            cv2.putText(frame, "ISL LIVE DETECTOR | Press Q to quit",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("ISL Live Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_detect()