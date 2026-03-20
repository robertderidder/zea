"""Flow matching / Rectified Flow generative models."""

from __future__ import annotations

import keras
from keras import ops

from zea.backend import _import_tf
from zea.func.tensor import split_seed
from zea.internal.registry import model_registry
from zea.models.dense import get_time_conditional_dense_network
from zea.models.generative import DeepGenerativeModel
from zea.models.unet import get_time_conditional_unetwork
from zea.models.utils import LossTrackerWrapper

tf = _import_tf()


@model_registry(name="flow")
class FlowModel(DeepGenerativeModel):
    """Implementation of a Flow Matching / Rectified Flow generative model."""

    def __init__(
        self,
        input_shape,
        network_name="unet_time_conditional",
        network_kwargs=None,
        name="flow_model",
        ema_val=0.999,
        **kwargs,
    ):
        """Initialize a Flow Matching / Rectified Flow model.

        Args:
            input_shape: Shape of the input data.
            network_name: Name of the network architecture.
            network_kwargs: Additional keyword arguments for the network.
            name: Name of the model.
            ema_val: Exponential moving average decay.
            **kwargs: Additional arguments.
        """
        super().__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.network_name = network_name
        self.network_kwargs = network_kwargs or {}
        self.ema_val = ema_val

        if network_name == "unet_time_conditional":
            self.network = get_time_conditional_unetwork(
                input_shape=input_shape, **self.network_kwargs
            )
        elif network_name == "dense_time_conditional":
            self.network = get_time_conditional_dense_network(
                input_shape=input_shape, **self.network_kwargs
            )
        else:
            raise ValueError(f"Unknown network name: {network_name}")

        self.ema_network = keras.models.clone_model(self.network)
        self.ema_network.trainable = False

        self.loss_tracker = LossTrackerWrapper("loss")

    def call(self, inputs, time, training=False, network=None, **kwargs):
        """Calls the velocity network."""
        if network is None:
            network = self.network if training else self.ema_network

        return network(inputs, time=time, training=training, **kwargs)

    def sample(self, n_samples=1, n_steps=50, method="euler", seed=None, **kwargs):
        """Sample from the flow model using standard ODE solvers.

        Args:
            n_samples: Number of samples to generate.
            n_steps: Number of integration steps.
            method: ODE solver method (currently only euler is implemented).
            seed: Random seed.
        """
        seed, seed1 = split_seed(seed, 2)
        x = keras.random.normal(shape=(n_samples, *self.input_shape), seed=seed1)
        
        step_size = 1.0 / n_steps
        for i in range(n_steps):
            t = ops.ones((n_samples, 1)) * (i / n_steps + step_size / 2) # Midpoint or start depending on solver
            t = ops.cast(t, x.dtype)
            v = self(x, time=t, training=False)
            x = x + v * step_size
            
        return x

    def train_step(self, data):
        # Unpack data
        if isinstance(data, dict):
            if "images" in data:
                data = data["images"]
            elif "image" in data:
                data = data["image"]
        elif isinstance(data, tuple):
            data = data[0]

        batch_size = ops.shape(data)[0]
        
        # 1. Sample t ~ U[0, 1]
        t = keras.random.uniform((batch_size, 1), minval=0.0, maxval=1.0, dtype=data.dtype)
        
        # 2. Sample x_0 ~ N(0, I)
        x_0 = keras.random.normal(ops.shape(data), dtype=data.dtype)
        
        # 3. Compute x_t = (1-t)x_0 + t x_1 (where data is x_1)
        # Reshape t for broadcasting
        t_broadcast = ops.reshape(t, (batch_size,) + (1,) * (len(self.input_shape)))
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * data
        
        # 4. Target vector field v_t = x_1 - x_0
        v_target = data - x_0
        
        with tf.GradientTape() as tape:
            # Predict vector field
            v_pred = self.network(x_t, time=t, training=True)
            loss = ops.mean(ops.square(v_pred - v_target))
            
        # Update weights
        trainable_vars = self.network.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update EMA
        for weight, ema_weight in zip(
            self.network.weights, self.ema_network.weights
        ):
            ema_weight.assign(
                self.ema_val * ema_weight + (1 - self.ema_val) * weight
            )
            
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}
        
    def test_step(self, data):
        if isinstance(data, dict):
            if "images" in data:
                data = data["images"]
            elif "image" in data:
                data = data["image"]
        elif isinstance(data, tuple):
            data = data[0]

        batch_size = ops.shape(data)[0]
        t = keras.random.uniform((batch_size, 1), minval=0.0, maxval=1.0, dtype=data.dtype)
        x_0 = keras.random.normal(ops.shape(data), dtype=data.dtype)
        
        t_broadcast = ops.reshape(t, (batch_size,) + (1,) * (len(self.input_shape)))
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * data
        v_target = data - x_0
        
        v_pred = self(x_t, time=t, training=False)
        loss = ops.mean(ops.square(v_pred - v_target))
        
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "network_name": self.network_name,
                "network_kwargs": self.network_kwargs,
                "ema_val": self.ema_val,
            }
        )
        return config
