"""
Real Image Recognition Model using Transfer Learning
Uses ResNet50 pre-trained on ImageNet, fine-tuned for museum artifacts
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from pathlib import Path
import numpy as np


class ArtifactRecognitionModel:
    """
    Image recognition model for museum artifacts
    Uses transfer learning with ResNet50
    """
    
    def __init__(self, num_classes=3, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Replace final layer for our number of artifact classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Load trained weights if available
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded model from {model_path}")
        else:
            print("⚠️  Using pre-trained ResNet50 (not fine-tuned on artifacts yet)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load artifact metadata
        self.artifacts = self._load_artifacts()
    
    def _load_artifacts(self):
        """Load artifact database"""
        artifacts_path = Path(__file__).parent.parent / "data" / "sample_artifacts.json"
        with open(artifacts_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def predict(self, image_path, top_k=3):
        """
        Predict artifact from image
        
        Args:
            image_path: Path to image file or PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with artifact info and confidence
        """
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            artifact_idx = int(idx)
            if artifact_idx < len(self.artifacts):
                artifact = self.artifacts[artifact_idx]
                predictions.append({
                    "artifact_id": artifact["id"],
                    "artifact": artifact,
                    "confidence": float(prob),
                    "model": "ResNet50"
                })
        
        return predictions
    
    def extract_features(self, image_path):
        """
        Extract feature vector from image (for similarity search)
        
        Args:
            image_path: Path to image file
            
        Returns:
            Feature vector (numpy array)
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Remove final classification layer
        features_model = nn.Sequential(*list(self.model.children())[:-1])
        
        with torch.no_grad():
            features = features_model(img_tensor)
            features = features.squeeze().cpu().numpy()
        
        return features


class ModelTrainer:
    """
    Trainer for fine-tuning the model on custom artifact dataset
    """
    
    def __init__(self, num_classes=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001
        )
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def save_model(self, path):
        """Save trained model"""
        torch.save(self.model.state_dict(), path)
        print(f"✅ Model saved to {path}")


if __name__ == "__main__":
    # Test the model
    print("Testing Artifact Recognition Model...")
    
    model = ArtifactRecognitionModel()
    print(f"Model loaded on: {model.device}")
    print(f"Number of artifact classes: {model.num_classes}")
    
    # Test with a sample image (if available)
    test_image_path = Path(__file__).parent.parent / "data" / "test_image.jpg"
    if test_image_path.exists():
        predictions = model.predict(test_image_path, top_k=3)
        print("\nPredictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['artifact']['name']} - {pred['confidence']:.2%}")
    else:
        print("\n⚠️  No test image found. Add images to data/ folder to test.")
