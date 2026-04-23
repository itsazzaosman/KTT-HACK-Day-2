import time
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import onnxruntime as ort

app = FastAPI(title="Edge-AI Crop Diagnostics")

# Load the ONNX model you just exported
session = ort.InferenceSession("model.onnx")
CLASSES = ["healthy", "maize_rust", "maize_blight", "cassava_mosaic", "bean_spot"]

# Pre-scripted rationales to help the Extension Officer explain the diagnosis
RATIONALES = {
    "healthy": "No visible signs of pathogens. Leaf texture and coloration are uniform.",
    "maize_rust": "Detected characteristic circular to elongate, golden-brown pustules on the leaf surface.",
    "maize_blight": "Detected large, cigar-shaped necrotic lesions indicative of fungal blight.",
    "cassava_mosaic": "Detected severe chlorotic mottling and misshapen leaflets typical of CMD.",
    "bean_spot": "Detected dark, angular lesions characteristic of fungal leaf spot."
}

def preprocess_image(image_bytes):
    # Convert image to RGB and resize to the 224x224 your model expects
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    # Convert to numpy array and scale to [0, 1]
    img_data = np.array(img).astype(np.float32) / 255.0
    
    # Normalize using ImageNet means/stds (required since we used the 'DEFAULT' backbone)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data - mean) / std
    
    # PyTorch expects Channels-First format: (Batch, Channels, Height, Width)
    img_data = np.transpose(img_data, (2, 0, 1))
    return np.expand_dims(img_data, axis=0)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    start_time = time.time()
    
    # Read and process the uploaded image
    contents = await image.read()
    input_tensor = preprocess_image(contents)
    
    # Run the ONNX session
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)
    logits = outputs[0][0]
    
    # Apply Softmax to convert raw logits into percentages
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    # Extract predictions
    top_indices = np.argsort(probs)[::-1]
    best_label = CLASSES[top_indices[0]]
    
    top3_dict = {CLASSES[idx]: round(float(probs[idx]), 4) for idx in top_indices[:3]}
    latency = round((time.time() - start_time) * 1000, 2)
    
    return {
        "label": best_label,
        "confidence": round(float(probs[top_indices[0]]), 4),
        "top3": top3_dict,
        "latency_ms": latency,
        "rationale": RATIONALES[best_label]
    }