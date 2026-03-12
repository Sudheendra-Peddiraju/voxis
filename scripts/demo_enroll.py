from voxis.config import VoxISConfig
from voxis.embedding import ECAPAEmbedder
from voxis.enrollment import EnrollmentService
from voxis.storage import TemplateStore


def main() -> None:
    config = VoxISConfig()
    print(f"Using device: {config.device}")

    tenant_id = "tenant_alpha"
    enrollment_files = [
        "sample_1.flac",
        "sample_2.flac",
    ]

    embedder = ECAPAEmbedder(config)
    service = EnrollmentService(
        embedder=embedder,
        sample_rate=config.sample_rate,
        segment_duration_sec=2.0,
    )

    result = service.enroll(audio_paths=enrollment_files, tenant_id=tenant_id)

    print("\nEnrollment complete")
    print("-------------------")
    print(f"Segments used: {result.num_segments_used}")
    print(f"Segment duration: {result.segment_duration_sec:.1f} sec")
    print(f"Protected template shape: {result.protected_template.shape}")
    
    store = TemplateStore()
    store.upsert_template(
        user_id="user_1",
        tenant_id=tenant_id,
        protected_template=result.protected_template,
    )
    print("Template stored successfully.")
    
    template = store.get_template("user_1", tenant_id)
    print(template.protected_template.shape)
    
    store = TemplateStore()
    t = store.get_template("user_1", "tenant_alpha")
    print(t.protected_template[:5])


if __name__ == "__main__":
    main()