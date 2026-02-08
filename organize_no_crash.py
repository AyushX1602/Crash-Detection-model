"""
Organize no_crash clips into train/val/test dataset structure
"""

import os
import shutil
import random
from tqdm import tqdm

# Configuration
SOURCE_DIR = "no_crash_clips"
DATASET_DIR = "Balanced Accident Video Dataset"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def organize_clips():
    """Organize clips into train/val/test folders"""
    
    print("="*60)
    print("ORGANIZING NO_CRASH CLIPS INTO DATASET")
    print("="*60)
    
    # Create folders if they don't exist
    for split in ['train', 'val', 'test']:
        folder = os.path.join(DATASET_DIR, split, 'no_crash')
        os.makedirs(folder, exist_ok=True)
    
    # Get all clips
    clips = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.mp4')]
    
    print(f"Total clips found: {len(clips)}")
    
    if len(clips) == 0:
        print("❌ No clips found! Run split_videos.py first.")
        return
    
    # Shuffle for random distribution
    random.shuffle(clips)
    
    # Calculate split points
    total = len(clips)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    # Split clips
    train_clips = clips[:train_end]
    val_clips = clips[train_end:val_end]
    test_clips = clips[val_end:]
    
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_clips)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val:   {len(val_clips)} ({VAL_RATIO*100:.0f}%)")
    print(f"  Test:  {len(test_clips)} ({TEST_RATIO*100:.0f}%)")
    
    # Copy files
    print("\n📁 Copying files...")
    
    for clip in tqdm(train_clips, desc="Train"):
        src = os.path.join(SOURCE_DIR, clip)
        dst = os.path.join(DATASET_DIR, 'train', 'no_crash', clip)
        shutil.copy2(src, dst)
    
    for clip in tqdm(val_clips, desc="Val  "):
        src = os.path.join(SOURCE_DIR, clip)
        dst = os.path.join(DATASET_DIR, 'val', 'no_crash', clip)
        shutil.copy2(src, dst)
    
    for clip in tqdm(test_clips, desc="Test "):
        src = os.path.join(SOURCE_DIR, clip)
        dst = os.path.join(DATASET_DIR, 'test', 'no_crash', clip)
        shutil.copy2(src, dst)
    
    print("\n" + "="*60)
    print("ORGANIZATION COMPLETE!")
    print("="*60)
    
    # Show final dataset structure
    print("\n📊 Final dataset structure:")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        for category in ['major', 'minor', 'moderate', 'no_crash']:
            folder = os.path.join(DATASET_DIR, split, category)
            if os.path.exists(folder):
                count = len([f for f in os.listdir(folder) if f.endswith('.mp4')])
                print(f"  {category:12s}: {count:4d} videos")
    
    print("\n✅ Ready for preprocessing!")
    print("\nNext steps:")
    print("1. Delete old preprocessed data: rmdir /s preprocessed")
    print("2. Preprocess: .\\venv\\Scripts\\python.exe preprocess_videos.py")
    print("3. Train: .\\venv\\Scripts\\python.exe train_improved.py")

if __name__ == "__main__":
    organize_clips()
