import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from cnn import create_cnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
train_dataset = datasets.ImageFolder(root=r"D:\animal_footprint_app\ml\animal-footprint\data\OpenAnimalTracks\cropped_imgs\train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Step 1️⃣: Initialize CNN for Training (with FC layer)
num_classes = len(train_dataset.classes)
cnn = create_cnn(num_classes, feature_extraction=False).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Step 2️⃣: Train the CNN
cnn.train()
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Step 3️⃣: Save CNN in Training Mode (with FC Layer)
print("Keys in trained CNN state_dict BEFORE saving:", cnn.state_dict().keys())
torch.save(cnn.state_dict(), "cnn_model.pth")
print("CNN model saved as cnn_model.pth")

# Step 4️⃣: Convert CNN to Feature Extraction Mode and Save Again
cnn = create_cnn(num_classes, feature_extraction=True).to(device)  # Convert model to feature extractor
cnn.load_state_dict(torch.load("cnn_model.pth", map_location=device), strict=False)  # Load trained weights

print("Keys in feature extraction CNN state_dict BEFORE saving:", cnn.state_dict().keys())

torch.save(cnn.state_dict(), "cnn_model_extracted.pth")  # Save model without FC layer

print("CNN model saved in feature extraction mode as cnn_model_extracted.pth")

# Step 5️⃣: Extract Features for PNN Training
features = []
labels = []

cnn.eval()  # Ensure model is in evaluation mode

with torch.no_grad():
    for images, label in train_loader:
        images = images.to(device)
        output = cnn(images).cpu()  # Extract CNN features
        features.extend(output.numpy())
        labels.extend(label.numpy())

# Step 6️⃣: Save CNN + PNN Feature Set
torch.save({
    'cnn': cnn.state_dict(),
    'features': features,
    'labels': labels,
    'class_names': train_dataset.classes
}, "cnn_pnn_model.pth")

print("CNN + PNN features saved as cnn_pnn_model.pth")
