import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from cnn import create_cnn
from pnn import PNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained CNN
num_classes = 18  # Adjust based on your classes
cnn = create_cnn(num_classes).to(device)
cnn.load_state_dict(torch.load("cnn_model.pth"))
cnn.eval()

# Remove Softmax layer
cnn.fc = torch.nn.Identity()

# Prepare data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=r"D:\animal_footprint_app\ml\animal-footprint\data\OpenAnimalTracks\cropped_imgs\train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# Extract features and train PNN
features = []
labels = []

for images, label in train_loader:
    images = images.to(device)
    with torch.no_grad():
        feature = cnn(images).cpu().numpy().flatten()
    features.append(feature)
    labels.append(label.item())

pnn = PNN()
pnn.fit(features, labels)

# Save the combined model
torch.save({'cnn': cnn.state_dict(), 'features': features, 'labels': labels}, "cnn_pnn_model.pth")
print("Combined CNN-PNN model saved as cnn_pnn_model.pth")
