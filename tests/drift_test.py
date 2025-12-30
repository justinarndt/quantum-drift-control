"""Unit tests for the Drift Orchestrator Physics Engine."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from src.drift.ou_process import OrnsteinUhlenbeckKernel

class DriftTest(absltest.TestCase):

    def test_ou_mean_reversion(self):
        """Verifies that the process reverts to the mean over time."""
        theta = 0.5
        sigma = 0.0  # Turn off noise to test deterministic drift
        dt = 0.1
        target_mean = 50.0
        start_val = 100.0
        
        kernel = OrnsteinUhlenbeckKernel(theta, sigma, dt)
        
        # Simulate one step
        key = jax.random.PRNGKey(0)
        x_prev = jnp.array([start_val])
        x_next = kernel.step(x_prev, target_mean, key)
        
        # Check direction
        self.assertLess(x_next[0], start_val, "Process should drift down towards mean")
        self.assertGreater(x_next[0], target_mean, "Process should not overshoot in one step")

    def test_stochastic_volatility(self):
        """Verifies that noise is injected when sigma > 0."""
        kernel = OrnsteinUhlenbeckKernel(theta=0.1, sigma=1.0, dt=0.01)
        key = jax.random.PRNGKey(42)
        
        x_prev = jnp.array([30.0])
        # Run two identical steps with different keys
        x_a = kernel.step(x_prev, 30.0, key)
        x_b = kernel.step(x_prev, 30.0, jax.random.split(key)[0])
        
        self.assertNotEqual(x_a[0], x_b[0], "Stochastic paths must diverge")

if __name__ == "__main__":
    absltest.main()