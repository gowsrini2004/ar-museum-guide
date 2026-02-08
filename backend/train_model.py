"""
Model training script for fine-tuning ResNet50 on custom artifacts
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json


def train_artifact_model(data_dir, num_epochs=10, batch_size=8, learning_rate=0.001):
    """
    Train ResNet50 on custom artifact dataset
    
    Args:
        data_dir: Path to training data directory
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        
    Returns:
        Dictionary with training results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    if len(dataset.classes) < 2:
        raise ValueError("Need at least 2 artifact classes to train")
    
    print(f"Found {len(dataset.classes)} artifact classes:")
    for idx, class_name in enumerate(dataset.classes):
        print(f"  {idx}: {class_name}")
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(dataset.classes))
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Training loop
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_dir = Path(__file__).parent.parent / "models"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / "artifact_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model (acc: {best_acc:.2f}%)")
    
    # Save class mapping
    class_mapping = {idx: class_name for idx, class_name in enumerate(dataset.classes)}
    mapping_path = Path(__file__).parent.parent / "models" / "class_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"\n{'='*40}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {model_path}")
    print(f"{'='*40}")
    
    return {
        "best_accuracy": best_acc,
        "num_classes": len(dataset.classes),
        "num_epochs": num_epochs,
        "history": history
    }


if __name__ == "__main__":
    # Train the model
    data_dir = Path(__file__).parent.parent / "data" / "training"
    
    if not data_dir.exists() or len(list(data_dir.iterdir())) < 2:
        print("❌ Error: Need at least 2 artifact folders in data/training/")
        print("\nExpected structure:")
        print("data/training/")
        print("  artifact_1/")
        print("    image1.jpg")
        print("    image2.jpg")
        print("  artifact_2/")
        print("    image1.jpg")
        print("    ...")
    else:
        results = train_artifact_model(str(data_dir))
        print(f"\n✅ Training completed with {results['best_accuracy']:.2f}% accuracy")
