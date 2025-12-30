"""Ornstein-Uhlenbeck Process Kernel for JAX.

This module provides the stochastic differential equation (SDE) solver 
required to model colored noise in superconducting qubits.
"""

import jax
import jax.numpy as jnp
from absl import logging

class OrnsteinUhlenbeckKernel:
    """JAX-compatible implementation of a mean-reverting stochastic process."""

    def __init__(self, theta: float, sigma: float, dt: float):
        """
        Args:
            theta: Mean reversion speed.
            sigma: Volatility (noise magnitude).
            dt: Time step size.
        """
        self.theta = theta
        self.sigma = sigma
        self.dt = dt

    @property
    def _diffusion_term(self):
        """Pre-calculates the diffusion variance for the time step."""
        return self.sigma * jnp.sqrt(self.dt)

    def step(self, x_prev: jnp.ndarray, mu: float, key: jnp.ndarray) -> jnp.ndarray:
        """Performs a single Euler-Maruyama integration step."""
        # Standard Brownian motion term dW
        noise = jax.random.normal(key, shape=x_prev.shape)
        
        # dX = theta * (mu - X) * dt + sigma * dW
        drift = self.theta * (mu - x_prev) * self.dt
        diffusion = self._diffusion_term * noise
        
        return x_prev + drift + diffusion

    def simulate_path(self, x0: jnp.ndarray, mus: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """Scans the SDE over a time horizon using JAX primitives."""
        def scan_fn(carry, mu_t):
            x_t, k = carry
            k, subk = jax.random.split(k)
            x_next = self.step(x_t, mu_t, subk)
            return (x_next, k), x_next

        _, trajectory = jax.lax.scan(scan_fn, (x0, key), mus)
        return trajectory