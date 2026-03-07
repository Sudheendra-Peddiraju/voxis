# VoxIS

VoxIS is a revocable voice identity system based on speaker embeddings and keyed orthogonal template protection.

## Current prototype
- ECAPA-TDNN embedding extraction
- Tenant-specific orthogonal transform
- Protected template generation: z = Rf
- Protected-space cosine verification

## Run
1. Create and activate `.venv`
2. Install torch + torchaudio
3. Install `requirements.txt`
4. Place `sample_1.wav` and `sample_2.wav` in repo root
5. Run:
   python scripts/demo_verify.py


## License

This repository is provided for research and demonstration purposes.
All rights reserved. Commercial use or redistribution is not permitted
without explicit permission from the author.