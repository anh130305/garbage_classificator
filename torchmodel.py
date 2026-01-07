import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------------------------
# 1️⃣ GPU setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# 2️⃣ Data loading
# -------------------------
data_dir = "dataset_split"  # replace with your dataset folder path

# Define transforms (resize, normalize like ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset   = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Found {len(train_dataset)} training images belonging to {len(train_dataset.classes)} classes.")
print(f"Found {len(val_dataset)} validation images belonging to {len(val_dataset.classes)} classes.")
print("Class mapping:", train_dataset.class_to_idx)

# -------------------------
# 3️⃣ Model setup (MobileNetV2)
# -------------------------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
num_classes = len(train_dataset.classes)

# Replace the classifier for your dataset
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# -------------------------
# 4️⃣ Loss and optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# 5️⃣ Training loop
# -------------------------
num_epochs = 10  # adjust as needed

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
# Save the model's parameters (state_dict)
torch.save(model.state_dict(), "mobilenetv2_trash_model.pth")
print("Model saved as mobilenetv2_trash_model.pth")


print("Training complete.")
