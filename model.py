import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import os

print("🚀 Starting Rapid Train & ONNX Export...")

# Setup & Data Loading
DATA_DIR = "dataset_output/mini_plant_set"
BATCH_SIZE = 32
NUM_CLASSES = 5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
test_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# Model Initialization (MobileNetV3-Small FP32)
model = models.mobilenet_v3_small(weights='DEFAULT')
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training Loop (3 Epochs for speed)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch():
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

for epoch in range(3):
    train_one_epoch()
    print(f"✅ Epoch {epoch+1} complete.")

# Evaluation (Macro-F1)
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"🎯 Clean Test Macro-F1: {macro_f1:.4f}")

# IMMEDIATE DIRECT EXPORT TO ONNX (Replacing the .pth save)
print("\nExporting directly to ONNX...")
model.to('cpu')
dummy_input = torch.randn(1, 3, 224, 224)
output_path = "model.onnx"

try:
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
    print(f"🎉 SUCCESS! '{output_path}' created.")
    print(f"📦 Final Size: {file_size:.2f} MB")
    
    if file_size < 10:
        print("✅ Task 2 Pass: Model is under the 10 MB limit.")

except Exception as e:
    print(f"❌ Export failed: {e}")