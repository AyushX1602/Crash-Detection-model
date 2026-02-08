"""
PREPROCESSING - OPTIMIZED FOR 6GB GPU
SEQUENCE_LENGTH = 16 (fits in memory with batch=8)
"""

import os
import numpy as np
import cv2
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle

DATASET_PATH = 'D:/programs vs/crash model/Balanced Accident Video Dataset'
OUTPUT_DIR = 'D:/programs vs/crash model/preprocessed'
SEQUENCE_LENGTH = 16  # ✅ Optimized for 6GB GPU
IMG_SIZE = 224
BATCH_SIZE = 25  # Smaller batches to avoid OOM during concatenation

def extract_frames(video_path, seq_len=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None
    
    indices = np.linspace(0, total - 1, seq_len, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        else:
            frames.append(frames[-1] if frames else np.zeros((img_size, img_size, 3), dtype=np.uint8))
    
    cap.release()
    while len(frames) < seq_len:
        frames.append(frames[-1])
    
    return np.array(frames[:seq_len], dtype=np.float32) / 255.0


def preprocess_split(split_name):
    """Memory-efficient preprocessing - saves incrementally"""
    data_path = os.path.join(DATASET_PATH, split_name)
    categories = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)
    
    print(f"\n📁 Processing {split_name} (SEQUENCE_LENGTH={SEQUENCE_LENGTH})...")
    
    # Collect all paths
    all_video_paths = []
    all_labels = []
    
    for cat in categories:
        cat_path = os.path.join(data_path, cat)
        videos = [f for f in os.listdir(cat_path) if f.endswith('.mp4')]
        for video in videos:
            all_video_paths.append(os.path.join(cat_path, video))
            all_labels.append(label_encoder.transform([cat])[0])
    
    total_videos = len(all_video_paths)
    print(f"  Total videos: {total_videos}")
    
    # Process in batches and save incrementally
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved_files_X = []
    saved_files_y = []
    
    batch_num = 0
    for batch_start in tqdm(range(0, total_videos, BATCH_SIZE), desc=f"  Batches"):
        batch_end = min(batch_start + BATCH_SIZE, total_videos)
        batch_frames = []
        batch_labels = []
        
        for i in range(batch_start, batch_end):
            frames = extract_frames(all_video_paths[i])
            if frames is not None:
                batch_frames.append(frames)
                batch_labels.append(all_labels[i])
        
        if batch_frames:
            X_batch = torch.tensor(np.array(batch_frames), dtype=torch.float32)
            X_batch = X_batch.permute(0, 1, 4, 2, 3)
            y_batch = torch.tensor(batch_labels, dtype=torch.long)
            
            # Save each batch separately
            batch_file_X = os.path.join(OUTPUT_DIR, f'{split_name}_X_batch{batch_num}.pt')
            batch_file_y = os.path.join(OUTPUT_DIR, f'{split_name}_y_batch{batch_num}.pt')
            
            torch.save(X_batch, batch_file_X)
            torch.save(y_batch, batch_file_y)
            
            saved_files_X.append(batch_file_X)
            saved_files_y.append(batch_file_y)
            
            del batch_frames, X_batch, y_batch
            batch_num += 1
    
    # Concatenate all saved batches (one at a time)
    print(f"  Concatenating {len(saved_files_X)} batches...")
    all_X = []
    all_y = []
    
    for file_X, file_y in zip(saved_files_X, saved_files_y):
        all_X.append(torch.load(file_X))
        all_y.append(torch.load(file_y))
    
    X = torch.cat(all_X, dim=0)
    y = torch.cat(all_y, dim=0)
    
    # Save final files
    torch.save(X, os.path.join(OUTPUT_DIR, f'{split_name}_X.pt'))
    torch.save(y, os.path.join(OUTPUT_DIR, f'{split_name}_y.pt'))
    
    # Delete batch files
    for file_X, file_y in zip(saved_files_X, saved_files_y):
        os.remove(file_X)
        os.remove(file_y)
    
    if split_name == 'train':
        with open(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(label_encoder, f)
    
    print(f"  ✅ Saved: {split_name}_X.pt ({X.shape})")
    return X.shape


if __name__ == "__main__":
    print("="*60)
    print("🔄 PREPROCESSING (OPTIMIZED FOR 6GB GPU)")
    print("="*60)
    print(f"   SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
    print(f"   This will fit in 6GB VRAM with batch=8")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        preprocess_split(split)
    
    print("\n✅ PREPROCESSING COMPLETE!")
    print(f"   Expected shapes: (N, {SEQUENCE_LENGTH}, 3, 224, 224)")
    print(f"   Now run: python train_improved.py")
