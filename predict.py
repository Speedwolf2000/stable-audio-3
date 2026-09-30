from cog import BasePredictor, Input, Path
from stable_audio_3 import StableAudioModel
import torch

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = StableAudioModel.from_pretrained("medium")
        
    def predict(
        self,
        prompt: str = Input(description="Text prompt for audio generation"),
        duration: int = Input(description="Duration in seconds", default=30)
    ) -> Path:
        """Run a single prediction on the model"""
        audio = self.model.generate(prompt=prompt, duration=duration)
        
        output_path = "/tmp/output.wav"
        # Logic to save the raw tensor 'audio' to output_path goes here
        
        return Path(output_path)
