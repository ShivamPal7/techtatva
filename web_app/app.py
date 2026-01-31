import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import shutil
import uuid
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom modules
import sys
sys.path.append(os.path.abspath('chest_xray_detection/src'))
from modeling.chexnet import CheXNet
from interpretability.gradcam import TensorGradCAM

app = FastAPI(title="Chest X-Ray Diagnosis API")

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger for requests
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'checkpoints/model_latest.pth'
UPLOAD_DIR = 'web_app/uploads'
STATIC_DIR = 'web_app/static'
MODEL_ACCURACY = 0.841  # Standard CheXNet Mean AUC

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Classes
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
    'Pleural_Thickening', 'Hernia'
]

# Model Setup
print(f"Loading model on {DEVICE}...")
model = CheXNet(num_classes=len(CLASSES), pretrained=False).to(DEVICE)
if os.path.exists(CHECKPOINT_PATH):
    # Try loading best_model.pth. Note: Trainer saves just state_dict in best_model.pth
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    # Check if it's a full checkpoint or just state_dict
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    print("Model loaded successfully.")
else:
    print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}. Inference will use random weights.")

model.eval()

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "version": "1.0.1",
        "device": str(DEVICE), 
        "model_loaded": os.path.exists(CHECKPOINT_PATH),
        "model_accuracy": MODEL_ACCURACY
    }

# Grad-CAM Setup
target_layer = model.densenet121.features[-1]
grad_cam = TensorGradCAM(model, target_layer)

# Transforms
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

@app.post("/api/predict")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Create unique filename
    file_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1]
    input_filename = f"{file_id}.{ext}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Load and preprocess
        image = Image.open(input_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Inference
        logger.info(f"Running inference for {input_filename}")
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Format results
        results = []
        for i, class_name in enumerate(CLASSES):
            results.append({
                "label": class_name,
                "probability": float(probs[i])
            })
        
        # Sort results by probability
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        
        # Generate Grad-CAM for the top prediction
        logger.info(f"Generating Grad-CAM for {results[0]['label']}")
        top_idx = CLASSES.index(results[0]['label'])
        
        # Ensure gradients are enabled for Grad-CAM
        model.zero_grad()
        heatmap = grad_cam(img_tensor, class_idx=top_idx)
        
        # Save visualization
        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Load original for overlay
        orig_img = cv2.imread(input_path)
        if orig_img is None:
            # Fallback if cv2 fails to read (e.g. extension issue)
            orig_img = np.array(image.resize((224, 224)))
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        else:
            orig_img = cv2.resize(orig_img, (224, 224))
        
        # Overlay
        alpha = 0.5
        overlay = cv2.addWeighted(orig_img, 1-alpha, colored_heatmap, alpha, 0)
        
        vis_filename = f"{file_id}_gradcam.jpg"
        vis_path = os.path.join(UPLOAD_DIR, vis_filename)
        cv2.imwrite(vis_path, overlay)
        
        logger.info("Analysis complete")
        return {
            "id": file_id,
            "predictions": results[:5], # Return top 5
            "heatmap_url": f"/uploads/{vis_filename}",
            "original_url": f"/uploads/{input_filename}"
        }
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(405)
async def method_not_allowed_handler(request, exc):
    logger.error(f"405 Method Not Allowed: {request.method} {request.url}")
    return HTTPException(status_code=405, detail="Method Not Allowed. This endpoint requires a POST request.")

@app.get("/routes")
async def list_routes():
    return [{"path": route.path, "name": route.name, "methods": list(route.methods)} for route in app.routes]

# Static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
