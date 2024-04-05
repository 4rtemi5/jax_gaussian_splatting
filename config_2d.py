from dataclasses import dataclass


@dataclass
class Config:
    debug: bool = 0
    dtype: str = "float32"
    image_size: tuple[int] = (16, 16)
    target_size: int = 1024
    channels: int = 3
    num_samples: int = 4096
    num_epochs: int = 10
    steps_per_epoch: int = 1000
    learning_rate: float = 0.001
    image_file_name: str = "demo_image.png"
    target_loss: float = -50.0  # max psnr loss
    random_seed: int = 42