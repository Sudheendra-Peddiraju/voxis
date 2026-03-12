from dataclasses import dataclass
import torch

@dataclass
class VoxISConfig:
    sample_rate: int = 16000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    ecapa_savedir: str = "pretrained_models/ecapa_voxceleb"
    db_path: str = "data/voxis.db"
    enrollment_segment_duration_sec: float = 2.0
    verification_threshold: float = 0.65