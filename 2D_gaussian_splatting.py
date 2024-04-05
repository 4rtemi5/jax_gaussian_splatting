#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

# In[2]:
# import gc
# import time
from config_2d import Config

# from typing import List
from functools import partial

import jax

if Config().debug:
    jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)

import keras

# import keras_core as keras
import matplotlib as mpl
import numpy as np

from imax import transforms

from jax import numpy as jnp
from matplotlib import pyplot as plt
from PIL import Image

mpl.rcParams["savefig.pad_inches"] = 0
plt.style.use("dark_background")


def downcast_safe(x, dtype, margin=0):
    x = jnp.clip(x, jnp.finfo(dtype).min + margin, jnp.finfo(dtype).max - margin)
    return x.astype(dtype)


# @partial(jax.jit, static_argnames=["image_size", "channels", "xy"])
def generate_2D_gaussian_splatting(
    sigma,
    rho,
    coords,
    colors,
    image_size,
    channels,
    xy=None,
):
    dtype = rho.dtype
    batch_size = colors.shape[0]
    sigma_x = sigma[:, 0:1].reshape((batch_size, 1, 1))
    sigma_y = sigma[:, 1:2].reshape((batch_size, 1, 1))
    sigma_z = sigma[:, 2:3].reshape((batch_size, 1, 1))
    # sigma_z = sigma[:, 2:3].reshape((batch_size, 1, 1))
    # coords = coords[:, :2]

    eps = 1e-4

    # sigma = sigma.reshape((batch_size, 1, 3))[:, :, :2]
    rho = rho.reshape((batch_size, 1, 1)) + eps

    covariance = jnp.concatenate(
        [
            jnp.concatenate(
                [
                    sigma_x * sigma_x + eps,
                    sigma_y * sigma_x * rho,
                    sigma_z * sigma_x * rho,
                ],
                axis=-1,
            ),
            jnp.concatenate(
                [
                    sigma_x * sigma_y * rho,
                    sigma_y * sigma_y + eps,
                    sigma_z * sigma_y * rho,
                ],
                axis=-1,
            ),
            jnp.concatenate(
                [
                    sigma_x * sigma_z * rho,
                    sigma_y * sigma_z * rho,
                    sigma_z * sigma_z + eps,
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    ).astype("float32")

    # covariance = jax.vmap(lambda s, r: jnp.outer(s, s) * r)(sigma[:, None, :2], rho[:, None, :])
    # covariance = covariance.clip(min=1e-8) + (jnp.eye(2)[None, ...] * 1e-6)

    inv_covariance = jnp.linalg.inv(covariance)

    if xy is None:
        # Creating a batch-wise meshgrid using broadcasting
        x = jnp.arange(image_size[1], dtype="float32") / (image_size[1])
        x = jnp.tile(x[None, None, ...], (1, image_size[0], 1))
        y = jnp.arange(image_size[0], dtype="float32") / (image_size[1])
        y = jnp.tile(y[None, ..., None], (1, 1, image_size[1]))
    else:
        x = xy[:, 0][None, None, ...]
        y = xy[:, 1][None, None, ...]

    ones = jnp.ones_like(x)

    xy = jnp.stack([x, y, ones], axis=-1)
    xy = xy + coords[:, None, None, :3]

    z = jnp.einsum("b...i,b...ij,b...j->b...", xy, -0.5 * inv_covariance, xy)

    # _, covariance_log_det = jnp.linalg.slogdet(covariance + eps)
    covariance_log_det = jnp.log(jnp.linalg.det(covariance))

    kernel = (
        z - jnp.log(2 * jnp.pi) + 0.5 * covariance_log_det.reshape((batch_size, 1, 1))
    )

    kernel_max = kernel.max(axis=[-1, -2], keepdims=True)
    kernel = jnp.exp(kernel - kernel_max)
    kernel = downcast_safe(kernel, dtype, margin=128)

    kernel = jnp.tile(
        kernel[..., None],
        (1, 1, 1, channels + 1),
    )

    rgb_values_reshaped = colors[..., None, None, :]
    rgb_kernel = (kernel * rgb_values_reshaped).sum(axis=0, keepdims=True)

    return rgb_kernel  # final_image


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

    window = create_window(window_size, channels)

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
    SSIM_denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    SSIM = SSIM_numerator / SSIM_denominator

    return (1 - SSIM) / 2


def d_ssim_loss(img1, img2):
    return ssim(img1, img2).mean()


def l1_loss(pred, target):
    return jnp.abs(target - pred).mean()


def psnr(pred, target):
    psnr = -10.0 * jnp.log(((target - pred) ** 2).mean()) / jnp.log(10.0)
    return psnr


# Combined Loss
@jax.jit
def combined_loss(pred, target, lambda_param=0.5):
    return (1 - lambda_param) * l1_loss(pred, target) + lambda_param * d_ssim_loss(
        pred, target
    )


@partial(jax.jit, static_argnames=("samples"))
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
        self.sigma = self.add_weight(
            shape=(self.total_samples, 3),
            initializer="ones",
            trainable=True,
            dtype=self.params_dtype,
        )
        # self.sigma_x = self.add_weight(
        #     shape=(self.total_samples, 1),
        #     initializer="ones",
        #     trainable=True,
        #     dtype=self.params_dtype,
        # )
        # self.sigma_y = self.add_weight(
        #     shape=(self.total_samples, 1),
        #     initializer="ones",
        #     trainable=True,
        #     dtype=self.params_dtype,
        # )
        self.coords = self.add_weight(
            shape=(self.total_samples, 3),
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
            shape=(self.total_samples, 4),
            initializer="zeros",
            trainable=True,
            dtype=self.params_dtype,
        )

    def call(self, transform=None, training=False):
        transform = jnp.eye(4) if transform is None else transform
        alpha = keras.ops.sigmoid(jnp.array(self.alpha))
        mask = (alpha > (1 / 255)) | (keras.ops.sigmoid(jnp.array(self.rho)) < 1 / 255)
        if self.config.channels == 4:
            colors = keras.ops.concatenate([self.colors, alpha], axis=-1)
        else:
            colors = jnp.array(self.colors) * alpha

        return (
            keras.ops.sigmoid(jnp.array(self.rho)).astype(self.config.dtype),
            jnp.matmul(
                transform,
                jnp.concatenate(
                    [
                        keras.ops.sigmoid(jnp.array(-self.sigma)),
                        jnp.ones_like(alpha),
                    ],
                    axis=1,
                ).T,
            ).T.astype(self.config.dtype),
            # keras.ops.sigmoid(jnp.array(-self.sigma_x)).astype(self.config.dtype),
            # keras.ops.sigmoid(jnp.array(-self.sigma_y)).astype(self.config.dtype),
            jnp.matmul(
                transform,
                jnp.concatenate(
                    [
                        keras.ops.tanh(jnp.array(self.coords)) + 0.5,
                        jnp.ones_like(alpha),
                    ],
                    axis=1,
                ).T,
            ).T.astype(self.config.dtype),
            keras.ops.tanh(colors).astype(self.config.dtype),
            mask,
        )


class SplatterModel(keras.Model):
    def __init__(self, config, splatter=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.image_size = config.image_size
        self.n_points = min(self.image_size[0] * self.image_size[1] - 1, 128**2)
        self.rng = jax.random.key(config.random_seed + sum(self.image_size))
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
        intrinsics,
        transform,
        target,
        xy,
        training=False,
    ):
        pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            transform,
            training=training,
        )
        rho, sigma, coords, colors, mask = pred

        splatted = generate_2D_gaussian_splatting(
            sigma,
            rho,
            coords,
            colors,
            self.image_size,
            target.shape[-1],
            xy,
        )
        splatted, depth = splatted[..., :3], splatted[..., -1]
        # splatted = transforms.apply_transform(
        #     splatted,
        #     transform=intrinsics,
        #     mask_value=-1,
        #     depth=depth,
        #     intrinsic_matrix=jnp.eye(3),
        #     bilinear=True
        # )
        # splatted = jnp.where(
        #     splatted > 1.0,
        #     splatted / (jnp.max(splatted, axis=-1, keepdims=True)  + 1e-12),
        #     splatted,
        # )
        # loss = combined_loss(splatted[None, ..., :], target, lambda_param=0.5)
        loss = -psnr(splatted[..., :3].flatten(), target[..., :3].flatten())

        return loss, (splatted, mask[..., None], non_trainable_variables)

    def call(self, data, training=False):
        return self.splatter(data)

    @staticmethod
    @partial(jax.jit, static_argnames=["xy", "target", "n"])
    def sample_n(xy, target, n, rng):
        if n == xy.shape[0]:
            return xy, target
        rng, seed_y = jax.random.split(rng, 2)
        row_i = jax.random.choice(
            seed_y, xy.shape[0], shape=(n,), replace=False
        ).astype("int32")
        xy = jnp.take_along_axis(xy, row_i[..., None], axis=0)

        target = target.reshape((-1, target.shape[-1]))
        target = jnp.take_along_axis(target, row_i[..., None], axis=0)
        return xy, target, rng

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        (intrinsics, transform, target) = data

        # Get the gradient function.
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        x = jnp.arange(self.image_size[1], dtype="float32") / (self.image_size[1])
        x = jnp.tile(x[None, None, ...], (1, self.image_size[0], 1))
        y = jnp.arange(self.image_size[0], dtype="float32") / (self.image_size[1])
        y = jnp.tile(y[None, ..., None], (1, 1, self.image_size[1]))
        xy = jnp.stack((x, y), axis=-1)
        xy = xy.reshape((-1, 2))

        xy, target, self.rng = self.sample_n(xy, target, self.n_points, self.rng)

        # Compute the gradients.
        (loss, (splatted, mask, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            intrinsics,
            transform,
            target,
            # pose,
            xy,
            training=True,
        )

        # clean gradients
        grads = [downcast_safe(g, self.config.dtype) for g in grads]

        # Update trainable variables and optimizer variables.
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        new_inits = init_values(self.config.num_samples)
        mask = mask.astype("float32")

        cleaned_trainable_variables = []

        for old_vars, new_vars in zip(trainable_variables, new_inits):
            cleaned_trainable_variables.append(
                jnp.clip(
                    old_vars * mask + new_vars.astype(old_vars.dtype) * (1 - mask),
                    jnp.finfo(old_vars.dtype).min + 128,
                    jnp.finfo(old_vars.dtype).max - 128,
                )
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
            "target": target,
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

    def test_step(self, state, data):
        # Unpack the data.
        (intrinsics, transform, target) = data
        (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        ) = state

        pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            transform,
        )
        rho, sigma, coords, colors, mask = pred

        splatted = generate_2D_gaussian_splatting(
            sigma,
            rho,
            coords,
            colors,
            self.image_size,
            target.shape[-1],
            xy=None,
        )
        splatted, depth = splatted[..., :3], splatted[..., -1]

        loss = -psnr(splatted, target[..., :3])

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
            "target": target,
        }
        new_metrics_vars = loss_tracker_vars

        # Return metric logs and updated state variables.
        state = (
            trainable_variables,
            non_trainable_variables,
            new_metrics_vars,
        )
        return logs, state


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

        ax.imshow(jnp.clip(self.target / 255.0, 0.0, 1.0), aspect="auto")
        fig.savefig(f"./results/target.png")
        plt.clf()
        plt.close()

    def on_train_batch_end(self, batch, logs):
        logs.pop("target")
        logs.pop("splatted")
        if logs["loss"] <= self.target_loss:
            self.model.stop_training = True

    def on_test_batch_end(self, batch, logs):
        logs.pop("loss")
        self.target = logs.pop("target")
        self.splatted = logs.pop("splatted")

    def on_epoch_end(self, epoch, logs={}):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(5, 5)

        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(jnp.clip(self.target[0], 0.0, 1.0), aspect="auto")
        fig.savefig(f"./results/target.png")
        plt.clf()
        plt.close()

        image = self.splatted
        image = jnp.clip(image, 0.0, 1.0)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(5, 5)

        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(jnp.clip(image[0], 0.0, 1.0), aspect="auto")
        fig.savefig(f"./results/{self.prefix}{epoch}.png")
        plt.clf()
        plt.close()


# In[13]:

# geometric augmentations


def data_gen(image_in, target_size, seed=0):
    key = jax.random.PRNGKey(seed)
    augment = 0

    height, width, _ = image_in.shape

    original_intrinsics = np.array(
        [
            [1, 0, (width - 1) / 2.0, 0.0],
            [0, 1, (height - 1) / 2.0, 0.0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype="float32",
    )

    original_depth = np.ones_like(image_in[:, :, 0], dtype="float32")

    aug_target_intrinsics = np.array(
        [
            [1, 0, (width - 1) / (target_size[1] - 1) / 2.0, 0.0],
            [0, 1, (height - 1) / (target_size[0] - 1) / 2.0, 0.0],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1.0],
        ],
        dtype="float32",
    )

    target_intrinsics = np.array(
        [
            [1, 0, (target_size[1] - 1) / (width - 1) / 2.0, 0.0],
            [0, 1, (target_size[0] - 1) / (height - 1) / 2.0, 0.0],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1.0],
        ],
        dtype="float32",
    )

    while True:
        augment = False  # (augment + 1) % 2

        if augment:
            translation = transforms.translate(
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
            )

            rotation = transforms.rotate(rad=np.random.uniform(0.0, 2.0))

            transform = translation

            augmented_original = transforms.apply_transform(
                image_in.astype("float32"),
                transform,
                mask_value=-1,
                depth=-1,
                intrinsic_matrix=-1,
                bilinear=True,
            )
            augmented_original = Image.fromarray(np.uint8(augmented_original))

            # augmented_original = Image.fromarray(np.uint8(image_in))

            augmented_original = augmented_original.resize(target_size, Image.BILINEAR)
            image = np.array(augmented_original) / 255.0
            transform = np.linalg.inv(target_intrinsics) @ transform
            intrinsics = target_intrinsics

        else:
            image = Image.fromarray(np.uint8(image_in))
            image = image.resize(target_size, Image.BILINEAR)
            image = np.array(image) / 255.0
            transform = np.linalg.inv(target_intrinsics) @ np.eye(4)
            intrinsics = target_intrinsics

        yield intrinsics, transform, image[None, ...]


# data

config = Config()
splatter_weights = Splatter(config)

original_image_size = config.image_size

for i in range(50):
    new_image_size = (
        int(original_image_size[0] * 2**i),
        int(original_image_size[1] * 2**i),
    )
    print(f"\ncurrent image_size: {new_image_size}")

    config.image_size = new_image_size
    splatter = SplatterModel(config=config, splatter=splatter_weights)

    # splatter.image_size = new_image_size
    # time.sleep(3 * i)

    original_image = Image.open(config.image_file_name)
    # original_image = original_image.resize(new_image_size, Image.BILINEAR)
    if config.channels == 3:
        original_image = original_image.convert("RGB")
    elif config.channels == 4:
        original_image = original_image.convert("RGBA")
    original_array = np.array(original_image)
    # original_array = original_array / 255.0

    target = jnp.array(original_array).astype("uint8")

    image_callback = ImageCallback(
        target, target_loss=config.target_loss, prefix=f"{min(new_image_size)}_"
    )

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        config.learning_rate,
        config.num_epochs * config.steps_per_epoch,
    )
    optimizer = keras.optimizers.Adam(lr_schedule)
    # optimizer = keras.optimizers.SGD(lr_schedule, clipvalue=1.0)
    splatter.compile(optimizer, run_eagerly=False)

    splatter.fit(
        data_gen(target, new_image_size, seed=i),
        validation_data=data_gen(target, new_image_size, seed=i),
        validation_steps=1,
        epochs=config.num_epochs,
        steps_per_epoch=config.steps_per_epoch,
        shuffle=False,
        callbacks=[image_callback],
    )
    if new_image_size[0] >= config.target_size:
        print("Done training.")
        break


# display results:

rho, sigma, coords, colors, mask = splatter(transform=None, training=False)
trafo = transforms.rotate(rad=jnp.pi / 3)[:3, :3]

coords = jnp.matmul(trafo, coords.T).T
coords = coords[:, :2]

sigma = jnp.matmul(trafo, sigma.T).T

sigma_x, sigma_y = sigma[:, :1], sigma[:, 1:2]

final_image = generate_2D_gaussian_splatting(
    sigma,
    rho,
    coords,
    colors,
    image_size=(config.target_size * 2, config.target_size * 2),
    channels=3,
)

plt.imshow(np.array(final_image.astype("float32")))
plt.savefig(f"./final.png")

print("Done.")
