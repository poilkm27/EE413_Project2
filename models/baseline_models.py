import torch
import torch.nn as nn
from torchvision import models, transforms

def get_transforms():
    # Resize to 96x96 as required for architectural divisibility by 2^5
    return transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_baseline_model(model_name='resnet18', num_classes=100):
    if model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) # FIXED SYNTAX
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes) # FIXED SYNTAX
    return model
