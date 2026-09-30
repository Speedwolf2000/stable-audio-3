import os
from cog import BasePredictor, Input, Path, Secret
from huggingface_hub import login
from stable_audio_3 import StableAudioModel
import torch
import torchaudio


class Predictor(BasePredictor):
    def setup(self):
        """Initialize predictor state; gated model loads on first prediction."""
        self.model = None

    def predict(
        self,
        prompt: str = Input(description="Text prompt for audio generation"),
        duration: int = Input(description="Duration in seconds", default=30),
        hf_token: Secret = Input(
            description="Hugging Face access token (hf_...) with access to stabilityai/stable-audio-3-medium",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if self.model is None:
            token = hf_token.get_secret_value() if hf_token else os.environ.get("HF_TOKEN")
            if not token:
                raise ValueError(
                    "Please provide your Hugging Face token (hf_...) in the hf_token input field."
                )
            login(token=token)
            self.model = StableAudioModel.from_pretrained("medium")

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
