from dataclasses import dataclass, field


@dataclass
class Config:
    debug: bool = False
    dtype: str = "float16"
    image_size: tuple[int] = (16, 16)
    channels: int = 3
    num_samples: int = 2500
    num_epochs: int = 10
    steps_per_epoch: int = 100
    learning_rate: float = 0.01  # 0.01
    image_file_name: str = "demo_image.png"
    target_loss: float = 0.005
