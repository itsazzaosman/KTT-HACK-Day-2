import torch
import torch.nn as nn
from torchvision import models
import torch.onnx
import os

# 1. Reconstruct the standard FP32 architecture
def get_model(num_classes=5):
    # Use weights='DEFAULT' to get the high-accuracy pre-trained base
    model = models.mobilenet_v3_small(weights='DEFAULT')
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

model = get_model()
model.eval()

# 2. Export to ONNX
# We use standard FP32 because MobileNetV3-Small is already < 10MB!
output_path = "model.onnx"
dummy_input = torch.randn(1, 3, 224, 224)

try:
    print(f"Exporting standard model to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True, 
        opset_version=13,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Success! 'model.onnx' created.")
    print(f"📦 Final Size: {file_size:.2f} MB")
    
    if file_size < 10:
        print("🚀 Task 2 Pass: Model is under the 10 MB limit.")

except Exception as e:
    print(f"❌ Export failed: {e}")