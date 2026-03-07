from __future__ import annotations
import torch
import torchaudio

def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Normalize amplitude
    waveform = waveform / waveform.abs().max().clamp(min=1e-9)

    return waveform