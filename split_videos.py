"""
Split long traffic videos into short clips
Perfect for creating training samples from compilation videos
"""

import cv2
import os
from tqdm import tqdm

# Configuration
INPUT_DIR = "no crash"
OUTPUT_DIR = "no_crash_clips"
CLIP_LENGTH = 10  # seconds per clip
MIN_VIDEO_LENGTH = 5  # Skip videos shorter than 5 seconds

def get_video_info(video_path):
    """Get video properties"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        'fps': fps,
        'duration': duration,
        'width': width,
        'height': height,
        'frame_count': frame_count
    }

def split_video(video_path, output_dir, clip_length=10):
    """
    Split video into clips of specified length
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save clips
        clip_length: Length of each clip in seconds
    
    Returns:
        Number of clips created
    """
    
    # Get video info
    info = get_video_info(video_path)
    
    if info['duration'] < MIN_VIDEO_LENGTH:
        print(f"⏭️  Skipping {os.path.basename(video_path)} (too short: {info['duration']:.1f}s)")
        return 0
    
    # Calculate number of clips
    num_clips = int(info['duration'] / clip_length)
    
    if num_clips == 0:
        print(f"⏭️  Skipping {os.path.basename(video_path)} (shorter than clip length)")
        return 0
    
    print(f"\n📹 Processing: {os.path.basename(video_path)}")
    print(f"   Duration: {info['duration']:.1f}s | FPS: {info['fps']:.1f} | Resolution: {info['width']}x{info['height']}")
    print(f"   Will create: {num_clips} clips")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Setup video writer parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(info['fps'])
    
    clips_created = 0
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Clean filename (remove special chars)
    base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_'))[:50]
    
    for clip_idx in tqdm(range(num_clips), desc="Creating clips"):
        # Calculate start frame
        start_frame = clip_idx * clip_length * fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Output path
        output_path = os.path.join(output_dir, f"{base_name}_clip_{clip_idx:04d}.mp4")
        
        # Create writer
        out = cv2.VideoWriter(output_path, fourcc, fps, (info['width'], info['height']))
        
        # Write frames
        frames_to_write = int(clip_length * fps)
        frames_written = 0
        
        for _ in range(frames_to_write):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1
        
        out.release()
        
        # Only count if we wrote enough frames
        if frames_written > frames_to_write * 0.8:  # At least 80% of expected frames
            clips_created += 1
        else:
            # Delete incomplete clip
            if os.path.exists(output_path):
                os.remove(output_path)
    
    cap.release()
    
    print(f"   ✅ Created {clips_created} clips")
    
    return clips_created

def main():
    """Main execution"""
    
    print("="*60)
    print("VIDEO SPLITTING FOR NO_CRASH CLASS")
    print("="*60)
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Clip length: {CLIP_LENGTH} seconds")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all videos
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    videos = []
    
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(video_extensions):
            videos.append(os.path.join(INPUT_DIR, file))
    
    print(f"\nFound {len(videos)} videos to process\n")
    
    # Process each video
    total_clips = 0
    
    for video_path in videos:
        try:
            clips = split_video(video_path, OUTPUT_DIR, CLIP_LENGTH)
            total_clips += clips
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(video_path)}: {e}")
            continue
    
    print("\n" + "="*60)
    print("SPLITTING COMPLETE!")
    print("="*60)
    print(f"Total clips created: {total_clips}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nNext step: Run organize_no_crash.py to organize into dataset")

if __name__ == "__main__":
    main()
