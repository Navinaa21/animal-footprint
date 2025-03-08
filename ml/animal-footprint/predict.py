import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from cnn import create_cnn
from pnn import PNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved model checkpoint with weights_only=False
checkpoint = torch.load("cnn_pnn_model.pth", map_location=device, weights_only=False)

# Step 1️⃣: Load CNN in Feature Extraction Mode
cnn = create_cnn(18, feature_extraction=True).to(device)

# 🛠️ Explicitly set strict=False to avoid key mismatches
cnn.load_state_dict(torch.load("cnn_model_extracted.pth", map_location=device), strict=False)
cnn.eval()

# Step 2️⃣: Load PNN Features and Labels
features = np.array(checkpoint.get('features', []))  # Ensure it's a NumPy array
labels = np.array(checkpoint.get('labels', []))

# Step 3️⃣: Load Class Names
class_names = checkpoint.get('class_names', [])

if not class_names:
    raise ValueError("Error: 'class_names' not found in checkpoint. Make sure it's saved in main.py")

# Step 4️⃣: Initialize PNN and Fit with Saved Features
pnn = PNN()
pnn.fit(features, labels)

# Step 5️⃣: Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Extract CNN Features
    with torch.no_grad():
        feature = cnn(image).cpu().numpy().flatten()

    # Predict with PNN
    label_idx = pnn.predict(feature)
    
    if label_idx >= len(class_names):
        raise IndexError(f"Predicted index {label_idx} out of bounds for class names list.")

    predicted_class = class_names[label_idx]

    print(f"Predicted Class: {predicted_class}")

# Step 6️⃣: Test with an Image
image_path = r"D:\animal_footprint_app\ml\animal-footprint\data\OpenAnimalTracks\cropped_imgs\test\mink\181.jpg"
predict_image(image_path)
