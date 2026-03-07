from voxis.config import VoxISConfig
from voxis.embedding import ECAPAEmbedder
from voxis.pipeline import VoxISPipeline


def main() -> None:
    config = VoxISConfig()
    print(f"Using device: {config.device}")

    tenant_id = "tenant_alpha"

    sample_1 = "sample_1.flac" # speaker A
    sample_2 = "sample_2.flac" # speaker A
    sample_3 = "sample_3.flac" # speaker B

    embedder = ECAPAEmbedder(config)
    pipeline = VoxISPipeline(embedder)

    score_12 = pipeline.verify_pair(sample_1, sample_2, tenant_id)
    score_13 = pipeline.verify_pair(sample_1, sample_3, tenant_id)
    score_23 = pipeline.verify_pair(sample_2, sample_3, tenant_id)

    print("\nProtected-space cosine similarities")
    print("-----------------------------------")
    print(f"sample_1 vs sample_2 (same speaker):      {score_12:.4f}")
    print(f"sample_1 vs sample_3 (different speaker): {score_13:.4f}")
    print(f"sample_2 vs sample_3 (different speaker): {score_23:.4f}")


if __name__ == "__main__":
    main()