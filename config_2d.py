from dataclasses import dataclass


@dataclass
class Config:
    debug: bool = 0
    dtype: str = "float32"
    target_size: int = 1024
    # image_size: tuple[int] = (target_size, target_size)
    image_size: tuple[int] = (16, 16)
    channels: int = 3
    num_samples: int = 4096 #* 2
    num_epochs: int = 100
    steps_per_epoch: int = 10000
    learning_rate: float = 0.0001
    image_file_name: str = "demo_image.png"
    target_loss: float = -45.0  # max psnr loss
    random_seed: int = 42