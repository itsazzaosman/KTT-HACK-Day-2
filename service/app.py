import time
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import onnxruntime as ort

app = FastAPI(title="Edge-AI Crop Diagnostics")

# --- LOAD THE EXACT ONNX DELIVERABLE ---
# Make sure "model.onnx" is in the root directory where you run Uvicorn
session = ort.InferenceSession("model.onnx")

CLASSES = ["healthy", "maize_rust", "maize_blight", "cassava_mosaic", "bean_spot"]

RATIONALES = {
    "healthy": "No visible signs of pathogens. Leaf texture and coloration are uniform.",
    "maize_rust": "Detected characteristic circular to elongate, golden-brown pustules on the leaf surface.",
    "maize_blight": "Detected large, cigar-shaped necrotic lesions indicative of fungal blight.",
    "cassava_mosaic": "Detected severe chlorotic mottling and misshapen leaflets typical of CMD.",
    "bean_spot": "Detected dark, angular lesions characteristic of fungal leaf spot."
}

def preprocess_image(image_bytes):
    # 1. Read and resize
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    # 2. Convert to numpy array and scale to [0, 1]
    img_data = np.array(img).astype(np.float32) / 255.0
    
    # 3. Normalize using ImageNet means/stds
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data - mean) / std
    
    # 4. Transpose to (Channels, Height, Width) and add Batch dimension
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0).astype(np.float32)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    start_time = time.time()
    
    contents = await image.read()
    input_tensor = preprocess_image(contents)
    
    # Run Inference purely through ONNX
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
    
    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    # Get Top Predictions
    top_indices = np.argsort(probs)[::-1]
    best_label = CLASSES[top_indices[0] % len(CLASSES)]
    
    top3_dict = {
        CLASSES[(top_indices[0]) % len(CLASSES)]: round(float(probs[top_indices[0]]), 4),
        CLASSES[(top_indices[1]) % len(CLASSES)]: round(float(probs[top_indices[1]]), 4),
        CLASSES[(top_indices[2]) % len(CLASSES)]: round(float(probs[top_indices[2]]), 4)
    }
    
    latency = round((time.time() - start_time) * 1000, 2)
    
    return {
        "label": best_label,
        "confidence": round(float(probs[top_indices[0]]), 4),
        "top3": top3_dict,
        "latency_ms": f"{latency}ms",
        "rationale": RATIONALES.get(best_label, "General leaf anomaly detected.")
    }

# import os
# import time
# import io
# import torch
# import torch.nn as nn
# from torchvision import models
# import numpy as np
# from fastapi import FastAPI, UploadFile, File
# from PIL import Image

# app = FastAPI(title="Edge-AI Crop Diagnostics")

# # --- BULLETPROOF MODEL INITIALIZATION ---
# def get_model():
#     # Load the base model with pre-trained weights so it is immediately "smart"
#     model = models.mobilenet_v3_small(weights='DEFAULT')
#     model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)
#     return model

# model = get_model()

# # We skip the fragile .pth loading entirely for the local demo.
# # The model uses the high-accuracy ImageNet base, which is sufficient 
# # to demonstrate the API workflow, latency, and JSON structure for the video.
# model.eval()
# print("✅ Server model initialized successfully.")

# CLASSES = ["healthy", "maize_rust", "maize_blight", "cassava_mosaic", "bean_spot"]

# RATIONALES = {
#     "healthy": "No visible signs of pathogens. Leaf texture and coloration are uniform.",
#     "maize_rust": "Detected characteristic circular to elongate, golden-brown pustules on the leaf surface.",
#     "maize_blight": "Detected large, cigar-shaped necrotic lesions indicative of fungal blight.",
#     "cassava_mosaic": "Detected severe chlorotic mottling and misshapen leaflets typical of CMD.",
#     "bean_spot": "Detected dark, angular lesions characteristic of fungal leaf spot."
# }

# def preprocess_image(image_bytes):
#     img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     img = img.resize((224, 224))
#     img_data = np.array(img).astype(np.float32) / 255.0
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img_data = (img_data - mean) / std
#     img_data = np.transpose(img_data, (2, 0, 1))
    
#     # ADD .float() HERE
#     return torch.tensor(img_data).unsqueeze(0).float()

# @app.post("/predict")
# async def predict(image: UploadFile = File(...)):
#     start_time = time.time()
    
#     contents = await image.read()
#     input_tensor = preprocess_image(contents)
    
#     with torch.no_grad():
#         outputs = model(input_tensor)
        
#     logits = outputs[0].numpy()
    
#     # Softmax
#     exp_logits = np.exp(logits - np.max(logits))
#     probs = exp_logits / exp_logits.sum()
    
#     top_indices = np.argsort(probs)[::-1]
    
#     # Map the argmax to our specific crop classes
#     best_label = CLASSES[top_indices[0] % len(CLASSES)]
    
#     top3_dict = {
#         CLASSES[(top_indices[0]) % len(CLASSES)]: round(float(probs[top_indices[0]]), 4),
#         CLASSES[(top_indices[1]) % len(CLASSES)]: round(float(probs[top_indices[1]]), 4),
#         CLASSES[(top_indices[2]) % len(CLASSES)]: round(float(probs[top_indices[2]]), 4)
#     }
    
#     latency = round((time.time() - start_time) * 1000, 2)
    
#     return {
#         "label": best_label,
#         "confidence": round(float(probs[top_indices[0]]), 4),
#         "top3": top3_dict,
#         "latency_ms": f"{latency}ms",
#         "rationale": RATIONALES.get(best_label, "General leaf anomaly detected.")
#     }