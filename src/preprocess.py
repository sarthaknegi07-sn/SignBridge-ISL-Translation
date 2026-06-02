import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.keypoint_utils import pose_to_sequence, normalize_sequence

# Paths
POSES_DIR = "data/raw/poses"
CSV_PATH = "data/raw/iSign_v1.1.csv"
PROCESSED_DIR = "data/processed"
MAX_SAMPLES_PER_CLASS = 100  # limit per phrase to keep balanced

def load_dataset():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["text"])

    # Clean text
    df["text"] = df["text"].str.strip().str.lower()

    # Filter short/long phrases
    df = df[df["text"].str.split().str.len().between(2, 8)]

    # Get top N most common phrases
    top_phrases = df["text"].value_counts().head(50).index.tolist()
    df = df[df["text"].isin(top_phrases)]

    print(f"Found {len(df)} samples across {len(top_phrases)} phrases")
    return df

def build_sequences(df):
    X, y = [], []
    pose_files = os.listdir(POSES_DIR)
    pose_lookup = {f.replace(".pose", ""): f for f in pose_files if f.endswith(".pose")}

    class_counts = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing poses"):
        uid = row["uid"]
        text = row["text"]

        # Limit samples per class
        class_counts[text] = class_counts.get(text, 0)
        if class_counts[text] >= MAX_SAMPLES_PER_CLASS:
            continue

        # Find matching pose file
        video_id = uid.split("-")[0]
        if video_id not in pose_lookup:
            continue

        pose_path = os.path.join(POSES_DIR, pose_lookup[video_id])
        sequence = pose_to_sequence(pose_path)
        sequence = normalize_sequence(sequence)

        if sequence.sum() == 0:
            continue

        X.append(sequence)
        y.append(text)
        class_counts[text] += 1

    return np.array(X, dtype=np.float32), np.array(y)

def preprocess():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_dataset()
    X, y = build_sequences(df)

    print(f"\nTotal sequences: {len(X)}")
    print(f"Sequence shape: {X.shape}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save label encoder
    with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Save splits
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Classes: {list(le.classes_)}")
    print("\nPreprocessing complete! Files saved to data/processed/")

if __name__ == "__main__":
    preprocess()