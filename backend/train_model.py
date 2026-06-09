"""
Model training script for fine-tuning ConvNeXt-Base on custom artifacts
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json
from datetime import datetime


def train_artifact_model(data_dir, num_epochs=15, batch_size=4, learning_rate=0.0001, progress_callback=None):
    """
    Train ConvNeXt-Base on custom artifact dataset with Feature Caching
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Progress file path
    progress_file = Path(__file__).parent.parent / "data" / "training_progress.json"
    
    def update_progress(epoch, total_epochs, status="training", accuracy=0.0, loss=0.0):
        progress = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "status": status,
            "accuracy": round(accuracy, 2),
            "loss": round(loss, 4),
            "percent": int(((epoch) / total_epochs) * 100) if total_epochs > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f)
        if progress_callback:
            progress_callback(progress)

    # Initial progress
    update_progress(0, num_epochs, status="loading_data")

    # Data transforms with Advanced Augmentation
    train_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset twice with different transforms so validation gets clean images
    # and training gets augmented images — WITHOUT sharing the dataset object.
    train_full = datasets.ImageFolder(data_dir, transform=train_transform)
    val_full   = datasets.ImageFolder(data_dir, transform=val_transform)

    if len(train_full.classes) < 2:
        raise ValueError("Need at least 2 artifact classes to train")

    # Deterministic split — same indices applied to both datasets
    import random
    total = len(train_full)
    indices = list(range(total))
    random.seed(42)
    random.shuffle(indices)
    train_size = int(0.85 * total)
    train_indices = indices[:train_size]
    val_indices   = indices[train_size:]

    train_ds = torch.utils.data.Subset(train_full, train_indices)
    val_ds   = torch.utils.data.Subset(val_full,   val_indices)

    # Keep full_dataset reference for class info
    full_dataset = train_full

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Load pre-trained ConvNeXt-Base
    update_progress(0, num_epochs, status="building_model")
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

    # Fine-Tuning Strategy:
    # 1. Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # 2. Unfreeze last stage of features for fine-grained adaptation
    for param in model.features[7].parameters():
        param.requires_grad = True

    # 3. Rebuild classifier head with Dropout to prevent overfitting
    num_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],  # LayerNorm2d
        model.classifier[1],  # Flatten
        nn.Dropout(p=0.3),
        nn.Linear(num_features, len(full_dataset.classes))
    )
    model.to(device)

    # Loss and optimizer — different LRs for backbone vs head
    params = [
        {'params': model.features[7].parameters(), 'lr': learning_rate * 0.1},
        {'params': model.classifier.parameters(), 'lr': learning_rate}
    ]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training Loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
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
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        scheduler.step()
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Save history and report progress
        update_progress(epoch + 1, num_epochs, status="training", accuracy=val_acc, loss=val_loss)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': full_dataset.class_to_idx,
                'classes': full_dataset.classes
            }, Path(__file__).parent.parent / "models" / "artifact_model.pth")
            
            # Save class mapping separately for ML API
            with open(Path(__file__).parent.parent / "models" / "class_mapping.json", 'w') as f:
                json.dump({v: k for k, v in full_dataset.class_to_idx.items()}, f, indent=2)
    
    # Final cleanup
    update_progress(num_epochs, num_epochs, status="completed", accuracy=best_acc)
    
    print(f"\nTraining Complete! Best Accuracy: {best_acc:.2f}%")
    
    avg_acc = sum(history['val_acc']) / len(history['val_acc']) if history['val_acc'] else 0.0
    
    return {
        "best_accuracy": best_acc,
        "avg_accuracy": avg_acc,
        "num_classes": len(full_dataset.classes),
        "num_epochs": num_epochs,
        "history": history
    }


if __name__ == "__main__":
    # Train the model
    data_dir = Path(__file__).parent.parent / "data" / "training"
    
    if not data_dir.exists() or len(list(data_dir.iterdir())) < 2:
        print("[ERR] Error: Need at least 2 artifact folders in data/training/")
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
        print(f"\n[OK] Training completed with {results['best_accuracy']:.2f}% accuracy")
