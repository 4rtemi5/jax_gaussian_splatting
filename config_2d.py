from dataclasses import dataclass


@dataclass
class Config:
    debug: bool = 0
    dtype: str = "float32"
    image_size: tuple[int] = (16, 16)
    target_size: int = 1024
    channels: int = 4
    num_samples: int = 4096
    num_epochs: int = 30
    steps_per_epoch: int = 1000
    learning_rate: float = 0.01
    image_file_name: str = "demo_image.png"
    target_loss: float = -45.0  # max psnr loss
    random_seed: int = 42