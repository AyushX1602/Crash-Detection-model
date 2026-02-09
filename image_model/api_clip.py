"""
FastAPI endpoint for CLIP crash detection
Integrate this with your SOS system
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from crash_detector_clip import CrashDetectorCLIP
from PIL import Image
import io
from typing import Dict
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="Crash Detection API",
    description="CLIP-based crash detection for SOS reporting system",
    version="1.0.0"
)

# Initialize detector (loaded once at startup)
print("🔥 Loading CLIP detector...")
detector = CrashDetectorCLIP()
print("✅ API ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Crash Detection API is running",
        "model": "CLIP (openai/clip-vit-base-patch32)",
        "status": "ready"
    }

@app.post("/predict")
async def predict_crash(
    file: UploadFile = File(...),
    threshold: float = 0.5
) -> Dict:
    """
    Predict if uploaded image contains a crash
    
    Args:
        file: Image file (jpg, png)
        threshold: Confidence threshold (0-1), default 0.5
    
    Returns:
        JSON with prediction results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected image."
            )
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get prediction
        result = detector.predict(image, threshold=threshold)
        
        # Add file info
        result['filename'] = file.filename
        result['file_size_kb'] = len(contents) / 1024
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/detailed")
async def predict_crash_detailed(file: UploadFile = File(...)) -> Dict:
    """
    Get detailed prediction with all prompt scores
    Useful for debugging and tuning
    
    Args:
        file: Image file
    
    Returns:
        JSON with detailed scores for each prompt
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}"
            )
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get basic prediction
        result = detector.predict(image)
        
        # Get detailed scores
        detailed = detector.get_detailed_scores(image)
        
        return JSONResponse(content={
            'prediction': result,
            'detailed_scores': detailed,
            'filename': file.filename
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts")
async def get_prompts():
    """Get current prompts being used"""
    return {
        'crash_prompts': detector.crash_prompts,
        'normal_prompts': detector.normal_prompts,
        'total_prompts': len(detector.crash_prompts) + len(detector.normal_prompts)
    }

@app.post("/prompts/update")
async def update_prompts(
    crash_prompts: list[str] = None,
    normal_prompts: list[str] = None
):
    """
    Update prompts for better accuracy
    This allows you to tune the model without retraining!
    
    Args:
        crash_prompts: New crash detection prompts
        normal_prompts: New normal traffic prompts
    """
    detector.update_prompts(crash_prompts, normal_prompts)
    
    return {
        'message': 'Prompts updated successfully',
        'crash_prompts_count': len(detector.crash_prompts),
        'normal_prompts_count': len(detector.normal_prompts)
    }


# Usage example for integration
def integrate_with_sos_system():
    """
    Example integration with your SOS system
    
    When user uploads photo from SOS button:
    """
    example_code = """
    import requests
    
    # User uploads photo via SOS button
    files = {'file': open('user_photo.jpg', 'rb')}
    
    # Call crash detection API
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()
    
    if result['is_crash'] and result['confidence'] > 0.8:
        # High confidence crash - auto-verify
        dispatch_ambulance(location)
        alert_hospitals(severity)
        
    elif result['is_crash'] and result['confidence'] > 0.6:
        # Medium confidence - send to operator
        queue_for_manual_review(photo, result)
        
    else:
        # Low confidence or no crash - ask for more info
        request_additional_details(user_id)
    """
    return example_code


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 STARTING CRASH DETECTION API")
    print("="*70)
    print("\nEndpoints:")
    print("  GET  /          - Health check")
    print("  POST /predict   - Crash detection")
    print("  POST /predict/detailed - Detailed analysis")
    print("  GET  /prompts   - View current prompts")
    print("  POST /prompts/update - Update prompts")
    print("\nDocs: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
