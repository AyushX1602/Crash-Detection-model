"""
Test script for CLIP crash detector
Run this to verify the detector works correctly
"""

from crash_detector_clip import CrashDetectorCLIP
import os
from pathlib import Path

def test_detector():
    """Test the CLIP detector with sample images"""
    
    print("="*70)
    print("🧪 TESTING CLIP CRASH DETECTOR")
    print("="*70)
    
    # Initialize detector
    detector = CrashDetectorCLIP()
    
    # Test different threshold values
    thresholds = [0.4, 0.5, 0.6]
    
    # Check if test images exist
    test_dir = Path("classification_dataset/test")
    
    if not test_dir.exists():
        print("\n⚠️  Test directory not found!")
        print("   Please provide test images in: classification_dataset/test/")
        print("   Or test manually with: python crash_detector_clip.py <image_path>")
        return
    
    # Get sample images
    crash_dir = test_dir / "crash"
    normal_dir = test_dir / "no_crash"
    
    crash_images = list(crash_dir.glob("*.jpg"))[:5] if crash_dir.exists() else []
    normal_images = list(normal_dir.glob("*.jpg"))[:5] if normal_dir.exists() else []
    
    if not crash_images and not normal_images:
        print("\n⚠️  No test images found!")
        print("   Add some images to:")
        print("   - classification_dataset/test/crash/")
        print("   - classification_dataset/test/no_crash/")
        return
    
    print(f"\n📊 Testing on:")
    print(f"   Crash images: {len(crash_images)}")
    print(f"   Normal images: {len(normal_images)}")
    
    # Test crash images
    if crash_images:
        print(f"\n{'='*70}")
        print("🚗 TESTING CRASH IMAGES")
        print(f"{'='*70}")
        
        for img_path in crash_images:
            print(f"\n📸 {img_path.name}")
            result = detector.predict(str(img_path))
            
            print(f"   Decision: {result['decision']}")
            print(f"   Crash Prob: {result['crash_probability']:.1%}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Correct: {'✅' if result['is_crash'] else '❌'}")
    
    # Test normal images
    if normal_images:
        print(f"\n{'='*70}")
        print("🚙 TESTING NORMAL IMAGES")
        print(f"{'='*70}")
        
        for img_path in normal_images:
            print(f"\n📸 {img_path.name}")
            result = detector.predict(str(img_path))
            
            print(f"   Decision: {result['decision']}")
            print(f"   Normal Prob: {result['normal_probability']:.1%}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Correct: {'✅' if not result['is_crash'] else '❌'}")
    
    # Calculate accuracy
    print(f"\n{'='*70}")
    print("📈 THRESHOLD ANALYSIS")
    print(f"{'='*70}")
    
    for threshold in thresholds:
        correct = 0
        total = 0
        
        # Test crash images
        for img_path in crash_images:
            result = detector.predict(str(img_path), threshold=threshold)
            if result['is_crash']:
                correct += 1
            total += 1
        
        # Test normal images
        for img_path in normal_images:
            result = detector.predict(str(img_path), threshold=threshold)
            if not result['is_crash']:
                correct += 1
            total += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"\nThreshold {threshold}: {accuracy:.1f}% accuracy ({correct}/{total})")
    
    print(f"\n{'='*70}")
    print("✅ TESTING COMPLETE")
    print(f"{'='*70}")
    print("\n💡 Tips to improve accuracy:")
    print("   1. Adjust threshold (try 0.4, 0.5, 0.6)")
    print("   2. Modify prompts in crash_detector_clip.py")
    print("   3. Use detailed_scores() to debug individual images")


def test_single_image(image_path):
    """Test single image with detailed output"""
    
    print("="*70)
    print("🔍 DETAILED IMAGE ANALYSIS")
    print("="*70)
    
    detector = CrashDetectorCLIP()
    
    # Basic prediction
    result = detector.predict(image_path)
    
    print(f"\nImage: {image_path}")
    print(f"\n🎯 PREDICTION: {result['decision']}")
    print(f"   Crash Probability: {result['crash_probability']:.2%}")
    print(f"   Normal Probability: {result['normal_probability']:.2%}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Threshold: {result['threshold']}")
    print(f"   Inference Time: {result['inference_time_ms']:.1f}ms")
    
    # Detailed scores
    print(f"\n📊 DETAILED PROMPT SCORES:")
    print("-" * 70)
    
    detailed = detector.get_detailed_scores(image_path)
    
    print("\n🚨 Crash Prompts:")
    for key, value in detailed['crash_prompts'].items():
        print(f"   {value['score']:.3f} - {value['prompt']}")
    
    print("\n✅ Normal Prompts:")
    for key, value in detailed['normal_prompts'].items():
        print(f"   {value['score']:.3f} - {value['prompt']}")
    
    print(f"\n📈 Total Scores:")
    print(f"   Crash: {detailed['total_crash_score']:.3f}")
    print(f"   Normal: {detailed['total_normal_score']:.3f}")
    
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test single image with details
        test_single_image(sys.argv[1])
    else:
        # Run full test suite
        test_detector()
