from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from voxis.config import VoxISConfig
from voxis.embedding import ECAPAEmbedder
from voxis.enrollment import EnrollmentService
from voxis.storage import TemplateStore
from voxis.verification import VerificationService

app = FastAPI(title="VoxIS API", version="0.1.0")

config = VoxISConfig()

# Load shared backend objects once at startup/import time
embedder = ECAPAEmbedder(config)
template_store = TemplateStore(db_path=config.db_path)

enrollment_service = EnrollmentService(
    embedder=embedder,
    sample_rate=config.sample_rate,
    segment_duration_sec=config.enrollment_segment_duration_sec,
)

verification_service = VerificationService(
    embedder=embedder,
    template_store=template_store,
    sample_rate=config.sample_rate,
    threshold=config.verification_threshold,
)


def _save_upload_to_temp(upload: UploadFile, tmp_dir: Path) -> Path:
    """
    Save an uploaded file to a temporary directory and return the saved path.
    """
    filename = upload.filename or "uploaded_audio"
    safe_name = Path(filename).name
    out_path = tmp_dir / safe_name

    with out_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

    return out_path


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": config.device,
        "model": config.ecapa_source,
        "db_path": config.db_path,
    }


@app.post("/enroll")
def enroll(
    user_id: Annotated[str, Form(...)],
    tenant_id: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> dict:
    """
    Enroll a user from one enrollment audio file.
    The audio is internally split into fixed-length segments,
    embeddings are averaged, and the protected template is stored.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        audio_path = _save_upload_to_temp(file, tmp_dir)

        try:
            result = enrollment_service.enroll(
                audio_paths=[str(audio_path)],
                tenant_id=tenant_id,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Enrollment failed: {e}")

        template_store.upsert_template(
            user_id=user_id,
            tenant_id=tenant_id,
            protected_template=result.protected_template,
            model_name=config.ecapa_source,
            transform_version="v1",
        )

    return {
        "message": "Enrollment successful",
        "user_id": user_id,
        "tenant_id": tenant_id,
        "num_segments_used": result.num_segments_used,
        "segment_duration_sec": result.segment_duration_sec,
        "template_shape": list(result.protected_template.shape),
    }


@app.post("/verify")
def verify(
    user_id: Annotated[str, Form(...)],
    tenant_id: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
) -> dict:
    """
    Verify a probe audio file against the enrolled protected template.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        probe_path = _save_upload_to_temp(file, tmp_dir)

        try:
            result = verification_service.verify(
                user_id=user_id,
                tenant_id=tenant_id,
                probe_audio_path=str(probe_path),
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Verification failed: {e}")

    return {
        "user_id": result.user_id,
        "tenant_id": result.tenant_id,
        "score": round(result.score, 4),
        "threshold": result.threshold,
        "verified": result.verified,
        "embedding_dim": result.embedding_dim,
    }