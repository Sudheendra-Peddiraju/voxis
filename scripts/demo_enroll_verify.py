from voxis.config import VoxISConfig
from voxis.embedding import ECAPAEmbedder
from voxis.enrollment import EnrollmentService
from voxis.storage import TemplateStore
from voxis.verification import VerificationService


def main() -> None:
    config = VoxISConfig()
    print(f"Using device: {config.device}")

    tenant_id = "tenant_alpha"
    user_id = "user_1"

    enrollment_files = [
        "sample_1.flac",
        "sample_2.flac",
    ]

    same_speaker_probe = "sample_4.flac"
    different_speaker_probe = "sample_3.flac"

    embedder = ECAPAEmbedder(config)
    store = TemplateStore()

    enrollment_service = EnrollmentService(
        embedder=embedder,
        sample_rate=config.sample_rate,
        segment_duration_sec=2.0,
    )

    verification_service = VerificationService(
        embedder=embedder,
        template_store=store,
        sample_rate=config.sample_rate,
        threshold=0.65,
    )

    enroll_result = enrollment_service.enroll(
        audio_paths=enrollment_files,
        tenant_id=tenant_id,
    )

    store.upsert_template(
        user_id=user_id,
        tenant_id=tenant_id,
        protected_template=enroll_result.protected_template,
        model_name=config.ecapa_source,
        transform_version="v1",
    )

    print("\nEnrollment complete")
    print("-------------------")
    print(f"Segments used: {enroll_result.num_segments_used}")
    print(f"Protected template shape: {enroll_result.protected_template.shape}")

    same_result = verification_service.verify(
        user_id=user_id,
        tenant_id=tenant_id,
        probe_audio_path=same_speaker_probe,
    )

    diff_result = verification_service.verify(
        user_id=user_id,
        tenant_id=tenant_id,
        probe_audio_path=different_speaker_probe,
    )

    print("\nVerification results")
    print("--------------------")
    print(f"Same-speaker probe score:      {same_result.score:.4f}")
    print(f"Same-speaker verified:         {same_result.verified}")
    print(f"Different-speaker probe score: {diff_result.score:.4f}")
    print(f"Different-speaker verified:    {diff_result.verified}")
    print(f"Threshold:                     {same_result.threshold:.2f}")


if __name__ == "__main__":
    main()