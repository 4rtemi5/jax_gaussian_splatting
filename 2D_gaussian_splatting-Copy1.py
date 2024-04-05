#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
os.environ["JAX_DEBUG_NANS"] = "true"

# In[2]:
import gc
import time
from config_2d import Config

RUN_EAGERLY = Config().train_eagerly

from typing import List
from functools import partial

import jax
from jax.experimental import checkify
import keras_core as keras
import matplotlib as mpl
import numpy as np
from imax.project import (
    bilinear_sampler,
    cam2pixel,
    meshgrid,
    nearest_sampler,
    pixel2cam,
)
from jax import numpy as jnp
from matplotlib import pyplot as plt
from PIL import Image

mpl.rcParams["savefig.pad_inches"] = 0
plt.style.use("dark_background")

# keras.mixed_precision.set_global_policy("mixed_float16")


def generate_2D_gaussian_splatting(
    sigma_x,
    sigma_y,
    # sigma_z,
    rho,
    coords,
    colors,
    image_size,
    channels,
):
    dtype = rho.dtype
    batch_size = colors.shape[0]
    kernel_size = max(image_size)
    sigma_x = sigma_x.reshape((batch_size, 1, 1))
    sigma_y = sigma_y.reshape((batch_size, 1, 1))
    # sigma_z = sigma_z.reshape((batch_size, 1, 1))
    rho = rho.reshape((batch_size, 1, 1))

    covariance = jnp.stack(
        [
            jnp.stack([sigma_x**2, rho * sigma_x * sigma_y], axis=-1),
            jnp.stack([rho * sigma_x * sigma_y, sigma_y**2], axis=-1),
        ],
        axis=-2,
    ).astype("float32")

    # Check for positive semi-definiteness
    # determinant = (sigma_x**2) * (sigma_y**2) - (rho * sigma_x * sigma_y) ** 2
    # determinant = jax.lax.clamp(1e-6, determinant, jnp.inf)

    inv_covariance = jnp.linalg.inv(covariance + 1e-6)

    ax_batch = jnp.linspace(-1.0, 1.0, num=kernel_size, dtype="float32")[None, ...]

    # Expanding dims for broadcasting
    ax_batch_expanded_x = jnp.tile(ax_batch[..., None], (1, 1, kernel_size))
    ax_batch_expanded_y = jnp.tile(ax_batch[:, None, ...], (1, kernel_size, 1))

    # Creating a batch-wise meshgrid using broadcasting
    xx, yy = ax_batch_expanded_x, ax_batch_expanded_y

    xy = jnp.stack([xx, yy], axis=-1)
    xy = xy + coords[:, None, None, :]

    z = jnp.einsum("b...i,b...ij,b...j->b...", xy, -0.5 * inv_covariance, xy)
    kernel = jnp.exp(z) / (
        2
        * jnp.pi
        * jnp.sqrt(jnp.linalg.det(covariance + 1e-6) + 1e-6).reshape((batch_size, 1, 1))
        + 1e-6
    )

    kernel_max = kernel.max(axis=[-1,-2], keepdims=True)
    kernel_max = jnp.where(kernel_max == 0, jnp.ones_like(kernel_max), kernel_max)
    kernel_normalized = jnp.clip(kernel / kernel_max, jnp.finfo(dtype).min + 1, jnp.finfo(dtype).max - 1).astype(dtype)

    kernel_reshaped = jnp.reshape(
        jnp.tile(kernel_normalized, (1, channels, 1)),
        (batch_size * channels, kernel_size, kernel_size),
    )
    kernel_rgb = kernel_reshaped.reshape(batch_size, channels, kernel_size, kernel_size)

    # Calculating the padding needed to match the image size
    pad_h = int(image_size[0]) - kernel_size
    pad_w = int(image_size[1]) - kernel_size

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller or equal to the image size.")

    # Adding padding to make kernel size equal to the image size
    padding = (
        (0, 0),
        (0, 0),
        (pad_w // 2, pad_w // 2 + pad_w % 2),  # padding top and bottom
        (pad_h // 2, pad_h // 2 + pad_h % 2),  # padding left and right
    )

    kernel_rgb_padded = jnp.pad(kernel_rgb, padding, "constant")
    kernel_rgb_padded = jnp.transpose(kernel_rgb_padded, (0, 2, 3, 1))

    # Extracting shape information
    b, h, w, c = kernel_rgb_padded.shape

    # Create a batch of 2D affine matrices
    intrinsics_0 = jnp.eye(2)
    intrinsics_0 = jnp.concatenate(
        [intrinsics_0, jnp.array([[(w - 1) / 2], [(h - 1) / 2]]), jnp.zeros((2, 1))],
        axis=1,
    )
    intrinsics_1 = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    intrinsics = jnp.concatenate([intrinsics_0, intrinsics_1], axis=0)
    # intrinsics = jnp.tile(intrinsics[None, ...], (b,1,1))

    depth = jnp.ones(shape=(h, w))

    # Creating grid and performing grid sampling
    pixel_coords = meshgrid(h, w, dtype="float32")
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics[:3, :3])

    src_pixel_coords = cam2pixel(cam_coords, intrinsics)

    if channels == 4:
        mask_value = jnp.array([0.0, 0.0, 0.0, 0.0])
    else:
        mask_value = jnp.array([0.0, 0.0, 0.0])

    kernel_rgb_padded_translated = jax.vmap(
        lambda rgb: nearest_sampler(rgb, src_pixel_coords, mask_value=mask_value)
    )(kernel_rgb_padded.astype("float32"))

    rgb_values_reshaped = colors[..., None, None, :]

    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated

    if channels == 4:
        final_image_layers = jnp.concatenate(
            [
                final_image_layers[..., :3] * final_image_layers[..., :1],
                final_image_layers[..., :1],
            ],
            axis=-1,
        )

    final_image = final_image_layers.sum(axis=0)
    # final_image = jax.lax.clamp(0.0, final_image, max=1.0)
    return final_image


if not RUN_EAGERLY:
    generate_2D_gaussian_splatting = jax.jit(
        generate_2D_gaussian_splatting, static_argnames=["image_size", "channels"]
    )


def gaussian_kernel(window_size, sigma=1.0):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = jnp.linspace(-(window_size - 1) / 2.0, (window_size - 1) / 2.0, window_size)
    gauss = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(sigma))
    kernel = jnp.outer(gauss, gauss)
    return kernel / jnp.sum(kernel)


def create_window(window_size, channels):
    _2D_window = gaussian_kernel(window_size, 1.5)[None, None, ...]
    window = jnp.broadcast_to(_2D_window, (channels, 1, window_size, window_size))
    return window


def conv2d(inputs, kernel, padding, groups):
    out = jax.lax.conv_general_dilated(
        inputs,  # lhs = NCHW image tensor
        kernel,  # rhs = OIHW conv kernel tensor
        (1, 1),  # window strides
        "VALID",  # padding mode
        feature_group_count=groups,
    )
    out = jnp.pad(
        out, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )
    return out


def ssim(img1, img2, window_size=11):
    channels = img1.shape[-1]

    img1 = jnp.transpose(img1, (0, 3, 1, 2))
    img2 = jnp.transpose(img2, (0, 3, 1, 2))

    # Parameters for SSIM
    C1 = 0.01**2
    C2 = 0.03**2

    window = create_window(window_size, channels).astype(img1.dtype)

    mu1 = conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = conv2d(img2, window, padding=window_size // 2, groups=channels)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    )
    sigma2_sq = (
        conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    )
    sigma12 = (
        conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2
    )

    SSIM_numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    SSIM_denominator = jnp.clip(
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2), 1e-6, jnp.inf
    )
    SSIM = SSIM_numerator / SSIM_denominator

    return (1 - SSIM) / 2


def d_ssim_loss(img1, img2):
    return ssim(img1, img2).mean()


def l1_loss(pred, target):
    return jnp.abs(target - pred).mean()


# Combined Loss
def combined_loss(pred, target, lambda_param=0.5):
    return (1 - lambda_param) * l1_loss(pred, target) + lambda_param * d_ssim_loss(
        pred, target
    ) + 1e-12


if not RUN_EAGERLY:
    combined_loss = jax.jit(combined_loss)


def init_values(samples, dtype="float32"):
    rho = jnp.ones((samples, 1), dtype=dtype)
    sigma_x = jnp.ones((samples, 1), dtype=dtype)
    sigma_y = jnp.ones((samples, 1), dtype=dtype)
    coords = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)(
        (samples, 2)
    ).astype(dtype)
    alpha = jnp.ones((samples, 1), dtype=dtype)
    colors = jnp.zeros((samples, 3), dtype=dtype)
    return rho, sigma_x, sigma_y, coords, alpha, colors


if not RUN_EAGERLY:
    init_values = jax.jit(init_values, static_argnames=("samples"))


class Splatter(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.channels = int(config.image_size[-1])
        self.total_samples = self.config.num_samples
        self.params_dtype = "float32"

    def build(self):
        self.rho = self.add_weight(
            shape=(self.total_samples, 1),
            initializer="ones",
            trainable=True,
            dtype=self.params_dtype,
        )
        self.sigma_x = self.add_weight(
            shape=(self.total_samples, 1),
            initializer="ones",
            trainable=True,
            dtype=self.params_dtype,
        )
        self.sigma_y = self.add_weight(
            shape=(self.total_samples, 1),
            initializer="ones",
            trainable=True,
            dtype=self.params_dtype,
        )
        self.coords = self.add_weight(
            shape=(self.total_samples, 2),
            initializer=keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
            trainable=True,
            dtype=self.params_dtype,
        )
        self.alpha = self.add_weight(
            shape=(self.total_samples, 1),
            initializer="ones",
            trainable=True,
            dtype=self.params_dtype,
        )
        self.colors = self.add_weight(
            shape=(self.total_samples, 3),
            initializer="zeros",
            trainable=True,
            dtype=self.params_dtype,
        )

    def call(self, inputs, training=False):
        step = inputs[0]

        alpha = keras.ops.sigmoid(jnp.array(self.alpha))

        mask = alpha > 0.01
        if self.config.channels == 4:
            colors = keras.ops.concatenate([self.colors, alpha], axis=-1)
        else:
            colors = jnp.array(self.colors) * alpha

        return (
            keras.ops.tanh(jnp.array(self.rho)),#.astype(self.config.dtype),
            -keras.ops.tanh(jnp.array(self.sigma_x)),# .astype(self.config.dtype),
            -keras.ops.tanh(jnp.array(self.sigma_y)),# .astype(self.config.dtype),
            keras.ops.tanh(jnp.array(self.coords)),# .astype(self.config.dtype),
            keras.ops.tanh(colors),# .astype(self.config.dtype),
            mask,
        )


class SplatterModel(keras.Model):
    def __init__(self, config, splatter=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.image_size = config.image_size
        if splatter is None:
            self.splatter = Splatter(config)
        else:
            self.splatter = splatter
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        x,
        target,
        # pose,
        training=False,
    ):
        pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            training=training,
        )
        rho, sigma_x, sigma_y, coords, colors, mask = pred
        splatted = generate_2D_gaussian_splatting(
            sigma_x,
            sigma_y,
            rho,
            coords,
            colors,
            self.image_size,
            target.shape[-1],
        )
        splatted = splatted.astype("float32")
        # splatted = jnp.where(
        #     splatted > 1.0,
        #     splatted / (jnp.max(splatted, axis=-1, keepdims=True)  + 1e-12),
        #     splatted,
        # )
        loss = combined_loss(splatted[None, ..., :], target, lambda_param=0.5)
        return loss, (splatted, mask[..., None], non_trainable_variables)

    def call(self, data, training=False):
        return self.splatter(data)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        (step, target) = data

        # Get the gradient function.
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        # print([v.dtype for v in trainable_variables])
        # print([v.dtype for v in non_trainable_variables])
        # print(step.dtype)
        # print(target.dtype)

        # Compute the gradients.
        (loss, (splatted, mask, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            step,
            target,
            # pose,
            training=True,
        )

        # clean gradients
        # grads = [jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)) for g in grads]

        # Update trainable variables and optimizer variables.
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        new_inits = init_values(self.config.num_samples)

        cleaned_trainable_variables = []

        for old_vars, new_vars in zip(trainable_variables, new_inits):
            cleaned_trainable_variables.append(
                old_vars * mask + new_vars.astype(old_vars.dtype) * (1 - mask)
            )

        # Update metrics.
        loss_tracker_vars = metrics_variables[: len(self.loss_tracker.variables)]

        loss_tracker_vars = self.loss_tracker.stateless_update_state(
            loss_tracker_vars, loss
        )

        logs = {
            self.loss_tracker.name: self.loss_tracker.stateless_result(
                loss_tracker_vars
            ),
            "splatted": splatted,
        }
        new_metrics_vars = loss_tracker_vars

        # Return metric logs and updated state variables.
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state


# In[12]:


class ImageCallback(keras.callbacks.Callback):
    def __init__(self, target, target_loss=0.0, prefix=""):
        super().__init__()
        self.target = target
        self.target_loss = target_loss
        self.prefix = prefix
        os.makedirs("./results", exist_ok=True)

    def on_train_begin(self, logs=None):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(5, 5)

        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(jnp.clip(self.target, 0.0, 1.0), aspect="auto")
        fig.savefig(f"./results/target.png")
        plt.clf()
        plt.close()

    def on_train_batch_end(self, batch, logs):
        self.splatted = logs.pop("splatted")
        if logs["loss"] <= self.target_loss:
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs={}):
        image = self.splatted
        image = jnp.clip(image, 0.0, 1.0)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(5, 5)

        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(image, aspect="auto")
        fig.savefig(f"./results/{self.prefix}{epoch}.png")
        plt.clf()
        plt.close()


# In[13]:


# data

config = Config()
splatter_weights = Splatter(config)

original_image_size = config.image_size

for i in range(1, 15):
    new_image_size = (
        int(original_image_size[0] * i),
        int(original_image_size[1] * i),
    )
    print(f"current image_size: {new_image_size}")

    config.image_size = new_image_size
    splatter = SplatterModel(config=config, splatter=splatter_weights)

    # splatter.image_size = new_image_size
    # time.sleep(3 * i)

    original_image = Image.open(config.image_file_name)
    original_image = original_image.resize(new_image_size)
    if config.channels == 3:
        original_image = original_image.convert("RGB")
    elif config.channels == 4:
        original_image = original_image.convert("RGBA")
    original_array = jnp.array(original_image)
    original_array = original_array / 255.0
    width, height, _ = original_array.shape

    image_array = original_array
    target = jnp.array(image_array, dtype="float32")

    image_callback = ImageCallback(
        target, target_loss=config.target_loss, prefix=f"{min(new_image_size)}_"
    )

    def data_gen(target):
        step = 0
        data = target[None, ...]

        while True:
            step += 1
            yield jnp.array([int(str(i) + str(step))]), data

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        config.learning_rate,
        config.num_epochs * config.steps_per_epoch,
    )
    optimizer = keras.optimizers.Adam(lr_schedule, clipnorm=1.0)
    splatter.compile(optimizer, run_eagerly=RUN_EAGERLY)

    splatter.fit(
        data_gen(target),
        epochs=config.num_epochs,
        steps_per_epoch=config.steps_per_epoch,
        shuffle=False,
        callbacks=[image_callback],
    )
    # gc.collect()


# In[ ]:


# rho, sigma_x, sigma_y, coords, colors = splatter(
#     jnp.array([1.0]), training=False
# )

# print(rho.shape)

# final_image = generate_2D_gaussian_splatting(
#     sigma_x,
#     sigma_y,
#     # sigma_z,
#     rho,
#     coords,
#     colors,
#     img_size,
#     3
# )

# plt.imshow(target)
# plt.axis("off")
# plt.tight_layout()
# plt.show()

# plt.imshow(final_image[:, :, :3])
# plt.axis("off")
# plt.tight_layout()
# plt.show()


# In[ ]:
