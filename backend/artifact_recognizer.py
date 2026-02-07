"""
Simple artifact recognition using image similarity
For minimal prototype - will be replaced with trained model later
"""
import json
import random
from pathlib import Path


class SimpleArtifactRecognizer:
    """Simulates artifact recognition for prototype"""
    
    def __init__(self, data_path="data/sample_artifacts.json"):
        self.data_path = Path(data_path)
        self.artifacts = self._load_artifacts()
    
    def _load_artifacts(self):
        """Load artifact database"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def recognize(self, image_path=None):
        """
        Simulate artifact recognition
        In real implementation: run detection + recognition models
        For prototype: return random artifact with confidence
        """
        # Simulate recognition with varying confidence
        artifact = random.choice(self.artifacts)
        confidence = random.uniform(0.85, 0.98)
        
        return {
            "artifact_id": artifact["id"],
            "artifact": artifact,
            "confidence": confidence,
            "top_3": [
                {"id": artifact["id"], "confidence": confidence},
                {"id": (artifact["id"] % 3) + 1, "confidence": confidence - 0.15},
                {"id": ((artifact["id"] + 1) % 3) + 1, "confidence": confidence - 0.25}
            ]
        }
    
    def get_artifact_by_id(self, artifact_id):
        """Get artifact details by ID"""
        for artifact in self.artifacts:
            if artifact["id"] == artifact_id:
                return artifact
        return None


if __name__ == "__main__":
    # Test the recognizer
    recognizer = SimpleArtifactRecognizer()
    result = recognizer.recognize()
    print(f"Recognized: {result['artifact']['name']}")
    print(f"Confidence: {result['confidence']:.2%}")
