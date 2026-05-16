import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from baseline_models import get_baseline_model, get_transforms

def train_baseline_architecture(model_name='resnet18', epochs=5, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    transform = get_transforms()

    # هنا التعديل المهم: استخدام المجلد الجديد data_split
    try:
        train_dataset = datasets.ImageFolder(root='data_split/train', transform=transform)
        test_dataset = datasets.ImageFolder(root='data_split/test', transform=transform)
    except FileNotFoundError:
        print("Error: Please run the data splitting script first.")
        return

    print(f"Dataset aligned: {len(train_dataset)} train images, {len(test_dataset)} test images.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.classes)
    model = get_baseline_model(model_name, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Training {model_name} ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader.dataset):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nFixed Baseline {model_name} Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train_baseline_architecture(model_name='resnet18', epochs=5)
