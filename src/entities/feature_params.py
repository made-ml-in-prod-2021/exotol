from dataclasses import dataclass, field


@dataclass()
class FeatureParams:

    target: list

    features_and_transformers_map: str = field(
        default="configs/features.yaml"
    )
