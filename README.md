# SignBridge: Real-Time Indian Sign Language (ISL) Translation & Speech Generation System

## Project Overview
SignBridge is an AI-powered Indian Sign Language (ISL) translation system designed to bridge communication barriers for the deaf and hard-of-hearing community. The system captures sign language gestures in real time through a webcam, translates them into text using deep learning, and converts the translated output into natural-sounding speech using neural text-to-speech technology.

## What's Done So Far
- [x] **Project Environment Setup:** Created virtual environment and `requirements.txt`.
- [x] **Voice Module:** Configured `voice_config.py` and implemented `speaker.py` using `edge-tts`.
- [x] **Preprocessing Pipeline:** Developed `keypoint_utils.py` and `preprocess.py` to handle `.pose` files and sequence generation.
- [x] **Model Architecture:** Designed a **Bidirectional LSTM with Attention Mechanism** in `model.py`.
- [x] **Training & Evaluation:** Implemented `train.py`, `dataset.py`, and `evaluate.py`.
- [x] **Real-Time Interface:** Developed `live_detect.py` for webcam-based inference and voice output.
- [x] **Project Structure & Documentation:** Set up `.gitignore` and `README.md`.

## Key Techniques & Features
*   **MediaPipe Integration:** Uses Google's MediaPipe Hands for high-performance 3D keypoint extraction from 2D images.
*   **Bidirectional LSTM (Bi-LSTM):** Captures temporal dependencies by processing sign language sequences in both forward and backward directions.
*   **Attention Mechanism:** Learns to assign weights to different frames in a gesture, allowing the model to focus on the most "expressive" moments of a sign.
*   **Edge-TTS Integration:** Uses Microsoft's Neural Text-to-Speech engine for high-quality, natural-sounding Indian English voices (`en-IN-NeerjaNeural`).
*   **Sequential Smoothing:** Implements a rolling frame buffer (30 frames) to maintain smooth real-time predictions.
*   **Confidence Filtering:** Uses a threshold (default 90%) to ensure only high-confidence translations are spoken.

## Project Structure
```text
D:\ISL_PROJECT\
├── data/
│   ├── raw/                # Raw .pose and CSV files (iSign v1.1)
│   └── processed/          # Preprocessed .npy and encoder files
├── models/                 # Saved model weights (.pth) and metrics
├── src/
│   ├── preprocess.py       # Data cleaning and sequence building
│   ├── dataset.py          # PyTorch Custom Dataset
│   ├── model.py            # Bidirectional LSTM + Attention model
│   ├── train.py            # Model training pipeline
│   ├── evaluate.py         # Testing and performance reports
│   └── live_detect.py      # Real-time inference and voice output
├── utils/
│   └── keypoint_utils.py   # Keypoint extraction and normalization
├── voice/
│   ├── speaker.py          # TTS synthesis and playback
│   ├── voice_config.py     # Voice personality and settings
│   └── audio_cache/        # Cached MP3 phrases for fast playback
├── requirements.txt        # Project dependencies
└── README.md
```

## System Requirements
- Python 3.10+
- Webcam for real-time detection
- Internet connection (for initial voice synthesis and pre-caching)

## Setup and Usage

### 1. Installation
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install pose-format
```

### 2. Prepare Data
Ensure your `.pose` files are in `data/raw/poses/` and the `iSign_v1.1.csv` file is in `data/raw/`.
```bash
python src/preprocess.py
```

### 3. Training
```bash
python src/train.py
```

### 4. Live Translation
```bash
python src/live_detect.py
```

## Future Roadmap
- [ ] Support for facial expression analysis.
- [ ] Expansion to multi-phrase sentence construction.
- [ ] Offline TTS support.

## License
MIT License
