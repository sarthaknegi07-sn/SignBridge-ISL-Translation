import numpy as np
from pose_format import Pose

# Keypoint dimensions
NUM_FRAMES = 30          # frames per gesture sequence
NUM_KEYPOINTS = 126      # 21 left hand + 21 right hand = 42 points x 3 (x,y,z)

def read_pose_file(pose_path):
    """Read a .pose file and return numpy array"""
    with open(pose_path, "rb") as f:
        buffer = f.read()
    pose = Pose.read(buffer)
    return pose.body.data, pose.body.confidence

def extract_keypoints(pose_data, confidence, frame_idx):
    """Extract hand keypoints from a single frame"""
    try:
        frame = pose_data[frame_idx]
        # flatten all keypoints for this frame
        keypoints = frame.flatten()
        # pad or truncate to NUM_KEYPOINTS
        if len(keypoints) >= NUM_KEYPOINTS:
            return keypoints[:NUM_KEYPOINTS]
        else:
            padded = np.zeros(NUM_KEYPOINTS)
            padded[:len(keypoints)] = keypoints
            return padded
    except:
        return np.zeros(NUM_KEYPOINTS)

def pose_to_sequence(pose_path):
    """
    Convert a .pose file to a fixed-length sequence of keypoints
    Returns: numpy array of shape (NUM_FRAMES, NUM_KEYPOINTS)
    """
    try:
        pose_data, confidence = read_pose_file(pose_path)
        total_frames = len(pose_data)

        if total_frames == 0:
            return np.zeros((NUM_FRAMES, NUM_KEYPOINTS))

        # Sample or pad to NUM_FRAMES
        if total_frames >= NUM_FRAMES:
            # uniformly sample NUM_FRAMES from total frames
            indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        else:
            # repeat last frame to pad
            indices = list(range(total_frames))
            indices += [total_frames - 1] * (NUM_FRAMES - total_frames)

        sequence = []
        for idx in indices:
            kp = extract_keypoints(pose_data, confidence, idx)
            sequence.append(kp)

        return np.array(sequence, dtype=np.float32)

    except Exception as e:
        print(f"Error processing {pose_path}: {e}")
        return np.zeros((NUM_FRAMES, NUM_KEYPOINTS), dtype=np.float32)

def normalize_sequence(sequence):
    """
    Normalize keypoints to be scale and position invariant
    sequence shape: (NUM_FRAMES, NUM_KEYPOINTS)
    """
    mean = sequence.mean()
    std = sequence.std()
    if std == 0:
        return sequence
    return (sequence - mean) / std