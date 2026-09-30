import os

# Force huggingface_hub to read directly from the baked-in container cache
# without making unauthenticated HEAD requests to the gated repo at runtime
os.environ["HF_HUB_OFFLINE"] = "1"

from cog import BasePredictor, Input, Path
from stable_audio_3 import StableAudioModel
import torch
import torchaudio


class Predictor(BasePredictor):
    def setup(self):
        """Load the baked-in model weights into GPU memory on container boot."""
        self.model = StableAudioModel.from_pretrained("medium")

    def predict(
        self,
        prompt: str = Input(description="Text prompt for audio generation"),
        duration: int = Input(description="Duration in seconds", default=30, ge=1, le=180),
    ) -> Path:
        """Run a single prediction on the loaded model."""
        audio = self.model.generate(prompt=prompt, duration=duration)

        # Handle tuple return (sample_rate, tensor) or raw tensor
        sample_rate = getattr(self.model, "sample_rate", 48000)
        if isinstance(audio, tuple):
            if isinstance(audio[0], int):
                sample_rate, audio = audio
            else:
                audio, sample_rate = audio

        if audio.dim() == 3:
            audio = audio[0]
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)

        output_path = "/tmp/output.wav"
        torchaudio.save(output_path, audio.detach().cpu().float(), sample_rate)

        return Path(output_path)
