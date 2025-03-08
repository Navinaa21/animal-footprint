import torch.nn as nn
import torchvision.models as models

def create_cnn(num_classes, feature_extraction=False):
    """
    Creates a ResNet18 model.
    If feature_extraction=True, removes the final classification layer.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if feature_extraction:
        model.fc = nn.Identity()  # Remove final classification layer for feature extraction

    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Normal classification mode

    print("Model structure:", model)  # 👀 Debugging - Check if fc is removed

    return model
