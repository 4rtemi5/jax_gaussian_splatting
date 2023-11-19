import jax
from jax import numpy as jnp


def breakpoint_if_not_finite(x):
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.breakpoint()

    jax.lax.cond(is_finite, true_fn, false_fn, x)
