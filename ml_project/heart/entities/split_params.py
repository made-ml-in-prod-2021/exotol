from dataclasses import dataclass, field


@dataclass
class SplitParameters:

    random_seed: int = field(default=100500)
    val_size: float = field(default=0.1)
    shuffle: bool = field(default=True)

