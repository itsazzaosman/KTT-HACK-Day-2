import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import os

# 1. Setup & Data Loading
DATA_DIR = "dataset_output/mini_plant_set"
BATCH_SIZE = 32
NUM_CLASSES = 5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)
test_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# 2. Model Initialization (MobileNetV3-Small)
# Chosen for its small footprint and quantization-friendly architecture
model = models.mobilenet_v3_small(weights='DEFAULT')
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 3. Training Loop
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

# Execute 5-10 epochs for quick hackathon results
for epoch in range(5):
    train_one_epoch()
    print(f"Epoch {epoch+1} complete.")

# 4. Evaluation (Macro-F1)
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
print(f"Clean Test Macro-F1: {macro_f1:.4f}")

# 5. Quantization to INT8
# This step is critical for the < 10MB requirement
model.to('cpu')
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Calibrate with a few batches
with torch.no_grad():
    for i, (imgs, _) in enumerate(train_loader):
        if i > 5: break
        model(imgs)
torch.quantization.convert(model, inplace=True)

# Save the quantized model
torch.save(model.state_dict(), "model_quantized.pth")
print(f"Model size: {os.path.getsize('model_quantized.pth') / 1e6:.2f} MB")