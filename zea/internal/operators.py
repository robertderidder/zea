"""Measurement operators.

Handles task-dependent operations (A) and noises (n) to simulate a measurement y = Ax + n.

"""

import abc

import numpy as np
from keras import ops
import jax
import jax.numpy as jnp

from zea.internal.core import Object
from zea.internal.registry import operator_registry


class Operator(abc.ABC, Object):
    """Operator base class.

    Used to define a generatic operator for a specific task / forward model.

    Examples are denoising, inpainting, deblurring, etc.

    One can derive linear and non-linear operators from this class.

    - Linear operators: y = Ax + n
    - Non-linear operators: y = f(x) + n

    """

    @abc.abstractmethod
    def forward(self, data, *args, **kwargs):
        """Implements the forward operator A: x -> y."""
        raise NotImplementedError

    @abc.abstractmethod
    def transpose(self, data, *args, **kwargs):
        """Implements the transpose (or adjoint) of the operator A^T: y -> x."""
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        """String representation of the operator."""
        raise NotImplementedError


@operator_registry(name="identity")
class IdentityOperator(Operator):
    """Identity operator class."""

    def forward(self, data):
        return data

    def transpose(self, data):
        return data

    def __str__(self):
        return "y = x"


@operator_registry(name="inpainting")
class InpaintingOperator(Operator):
    """Inpainting operator class.

    Inpainting task is a linear operator that masks the data with a mask.

    Formally defined as:
        y = Ax + n, where A = I * M

    Note that this generally only is the case for min_val = 0.0.
    Since we implement the operator using `ops.where`.

    where I is the identity operator, M is the mask, and n is the noise.
    """

    def __init__(self, min_val=0.0, **kwargs):
        """Initialize the inpainting operator.

        Args:
            min_val: Minimum value for the mask.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.min_val = min_val

    def forward(self, data, mask):
        # return self.mask * data
        return ops.where(mask, data, self.min_val)

    def transpose(self, data, mask):
        # masking operation is diagonal --> A.T = A
        return self.forward(data, mask)

    def __str__(self):
        return "y = Ax + n, where A = I * M"


@operator_registry(name="fourier_blur")
class FourierBlurOperator(Operator):
    """Fourier-domain blurring operator class.

    Applies blurring by masking high frequencies in the Fourier domain.

    Formally defined as:
        y = F^(-1)(M * F(x))

    where F is the FFT, F^(-1) is the inverse FFT and M is the frequency mask.
    """

    def __init__(self, shape, cutoff_freq=0.5, smooth=True, **kwargs):
        """Initialize the Fourier blur operator.

        Args:
            shape: Shape of the input data (H, W), (H, W, C), or (B, H, W, C).
            cutoff_freq: Cutoff frequency as fraction of Nyquist frequency (0.0 to 1.0).
            smooth: If True, use Gaussian rolloff; otherwise use hard cutoff.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.cutoff_freq = cutoff_freq
        self.shape = shape

        # Precompute frequency mask
        self.freq_mask = self.make_lowpass_mask(shape=shape, cutoff_freq=cutoff_freq, smooth=smooth)

    def make_lowpass_mask(self, shape, cutoff_freq=0.1, smooth=True):
        """
        Create a low-pass Fourier mask of given shape.
        cutoff: relative frequency radius (0 < cutoff < 0.5)
        smooth: if True, use Gaussian rolloff
        """
        # Accept (H, W), (H, W, C) or (B, H, W, C)
        if len(shape) == 2:
            H, W = shape
        elif len(shape) >= 3:
            H, W = shape[-3], shape[-2]
        else:
            raise ValueError(f"Invalid shape {shape}. Expected (H, W), (H, W, C), or (B, H, W, C).")
        fy = np.fft.fftfreq(H)
        fx = np.fft.fftfreq(W)
        FX, FY = np.meshgrid(fx, fy)
        R = np.sqrt(FX**2 + FY**2)

        if smooth:
            sigma = cutoff_freq / np.sqrt(2 * np.log(2))
            mask = np.exp(-(R**2) / (2 * sigma**2))
        else:
            mask = (R < cutoff_freq).astype(np.float32)

        # Shift DC to top-left to match fft2 conventions
        # mask = np.fft.ifftshift(mask)
        return ops.convert_to_tensor(mask)

    def forward(self, data):
        """Apply Fourier-domain blurring.

        Args:
            data: Input tensor of shape (B, H, W, C)

        Returns:
            Blurred data tensor.
        """
        # Convert to float32 for FFT
        data_real = ops.cast(data, "float32")
        data_imag = ops.zeros_like(data_real)

        # fft2 calculates the 2D FFT on the last two dims, so we want
        # H, W to be at the end
        data_real = ops.transpose(data_real, (0, 3, 1, 2))
        data_imag = ops.transpose(data_imag, (0, 3, 1, 2))

        # Apply FFT - expects tuple (real, imag), returns tuple (real, imag)
        fft_real, fft_imag = ops.fft2((data_real, data_imag))

        # Apply frequency mask to both real and imaginary parts
        mask_real = ops.real(self.freq_mask)  # Extract real part of complex mask
        masked_fft_real = fft_real * mask_real
        masked_fft_imag = fft_imag * mask_real

        # Apply inverse FFT
        blurred_real, blurred_imag = ops.ifft2((masked_fft_real, masked_fft_imag))

        # transpose back to original shape
        blurred_real = ops.transpose(blurred_real, (0, 2, 3, 1))

        # Take real part (imaginary should be ~0 for real input)
        blurred_data = ops.cast(blurred_real, data.dtype)

        return blurred_data

    def transpose(self, data):
        """
        transpose = forward because A^* (F^{-1} M F)^* = (F^* M^* (F^{-1})^*) = F^{-1} M F = A
        i.e. this is a self-adjoint operator.
        """
        return self.forward(data)

    def __str__(self):
        return f"y = F^(-1)(M * F(x)) filter at {self.cutoff_freq}"


from zea.simulator import simulate_rf

@operator_registry(name="simulate_zea")
class ZeaSimulatorOperator(Operator):
    """
    Operator for simulating RF data from an image.
    The operator is initialized with fixed scan and probe parameters.
    The forward operation takes an image, computes scatterer positions and magnitudes,
    and simulates the RF data using the ultrasound simulator.
    """

    def __init__(self, scan, probe, shape, chunk_size = 1024):
        super().__init__()
        self.scan = scan
        self.probe = probe
        self.shape = shape
        self.positions = self.compute_scatterer_positions()
        self.chunk_size = chunk_size

    def compute_scatterer_positions(self):
        """
        Compute scatterer positions for each pixel in the image, using scan x/z limits and image shape.
        Returns an array of shape (n_pixels, 3) with [x, y, z] positions.
        """
        x_start, x_end = self.scan.xlims
        z_start, z_end = self.scan.zlims
        # Accept (H, W), (H, W, C) or (B, H, W, C)
        if len(self.shape) == 2:
            H, W = self.shape
        elif len(self.shape) >= 3:
            H, W = self.shape[-3], self.shape[-2]
        else:
            raise ValueError(f"Invalid shape {self.shape}. Expected (H, W), (H, W, C), or (B, H, W, C).")
        x = ops.linspace(x_start, x_end, H)
        z = ops.linspace(z_start, z_end, W)
        xv, zv = ops.meshgrid(x, z, indexing='ij')
        x_flat = ops.reshape(xv, [-1])
        z_flat = ops.reshape(zv, [-1])
        y_flat = ops.zeros_like(x_flat)
        positions = ops.stack([x_flat, y_flat, z_flat], axis=1)
        return positions
    
    def forward(self, image):
        """
        Simulates RF data from the input image.
        Each pixel is a scatterer, pixel value is magnitude.
        """
        n_frames, *img_shape = image.shape
        magnitudes = ops.reshape(image, (n_frames, -1))
        @jax.checkpoint
        def simulate_chunk_fn(chunk_positions, chunk_magnitudes):
            return simulate_rf(
                scatterer_positions=chunk_positions,
                scatterer_magnitudes=chunk_magnitudes,
                probe_geometry=self.probe.probe_geometry,
                apply_lens_correction=self.scan.apply_lens_correction,
                lens_thickness=self.scan.lens_thickness,
                lens_sound_speed=self.scan.lens_sound_speed,
                sound_speed=self.scan.sound_speed,
                n_ax=self.scan.n_ax,
                center_frequency=self.probe.center_frequency,
                sampling_frequency=self.probe.sampling_frequency,
                t0_delays=self.scan.t0_delays,
                initial_times=self.scan.initial_times,
                element_width=self.scan.element_width,
                attenuation_coef=self.scan.attenuation_coef,
                tx_apodizations=self.scan.tx_apodizations,
            )
        
        n_scatterers = self.positions.shape[0]
        n_chunks = n_scatterers // self.chunk_size

        # 2. Reshape into chunks: (N_chunks, chunk_size, ...)
        # Note: This assumes n_scatterers is divisible by self.chunk_size. 
        # If not, pad self.positions and magnitudes with zeros.
        n_chunks = n_scatterers // self.chunk_size
        remainder = n_scatterers % self.chunk_size
        if remainder == 0:
            n_padded = n_scatterers
            pad_width = 0
        else:
            pad_width = self.chunk_size-remainder
            n_padded = n_scatterers+pad_width

        padded_pos = jnp.pad(self.positions, ((0, pad_width), (0, 0)))
        padded_mag = jnp.pad(magnitudes, ((0, 0), (0, pad_width)))

        n_chunks = n_padded // self.chunk_size
        reshaped_pos = ops.reshape(padded_pos, (n_chunks, self.chunk_size, 3))
        reshaped_mag = ops.reshape(padded_mag, (n_frames, n_chunks, self.chunk_size))
        reshaped_mag = jnp.swapaxes(reshaped_mag, 0, 1)

        # 3. Define the scan carrier function
        def body_fn(carry, x):
            pos_c, mag_c = x
            # vmap over the n_frames dimension
            rf_chunk = jax.vmap(simulate_chunk_fn, in_axes=(None, 0))(pos_c, mag_c)
            return carry + rf_chunk, None

        # 4. Initialize total_rf and run the loop
        # You need to know the output shape of simulate_rf (e.g., [n_samples, n_channels])
        out_shape = (n_frames, 1, self.scan.n_tx, self.scan.n_el, self.scan.n_ax, 1) # adjust accordingly
        init_rf = jnp.zeros(out_shape)
        
        total_rf, _ = jax.lax.scan(body_fn, init_rf, (reshaped_pos, reshaped_mag))
        return total_rf
    
    def transpose(self, data):
        raise NotImplementedError("Transpose for SimulateOperator is not implemented.")

    def __str__(self):
        return f"SimulateOperator with scan n_tx={self.scan.n_tx}, probe center_freq={self.probe.center_frequency}"
    
from jaxus.rf_simulator import _get_vectorized_simulate_function
from jaxus.rf_simulator import simulate_rf_data
from jaxus.containers.waveform import get_pulse


@operator_registry(name="simulator_full")
class SimulatorFull(Operator):
    """
    Operator for simulating RF data from an image.
    The operator is initialized with fixed scan and probe parameters.
    The forward operation takes an image, computes scatterer positions and magnitudes,
    and simulates the RF data using the ultrasound simulator.
    """

    def __init__(self, scan, shape):
        super().__init__()
        self.scan = scan
        self.shape = shape
        self.positions = self.compute_scatterer_positions()

    def compute_scatterer_positions(self):
        """
        Compute scatterer positions for each pixel in the image, using scan x/z limits and image shape.
        Returns an array of shape (n_pixels, 3) with [x, y, z] positions.
        #TODO: Distinguish between scatterers, and from image
        """
        x_start, x_end = self.scan.xlims
        z_start, z_end = self.scan.zlims
        # Accept (H, W), (H, W, C) or (B, H, W, C)
        if len(self.shape) == 2:
            H, W = self.shape
        elif len(self.shape) >= 3:
            H, W = self.shape[-3], self.shape[-2]
        else:
            raise ValueError(f"Invalid shape {self.shape}. Expected (H, W), (H, W, C), or (B, H, W, C).")
        x = ops.linspace(x_start, x_end, H)
        z = ops.linspace(z_start, z_end, W)
        xv, zv = ops.meshgrid(x, z, indexing='ij')
        x_flat = ops.reshape(xv, [-1])
        z_flat = ops.reshape(zv, [-1])
        y_flat = ops.zeros_like(x_flat)
        positions = ops.stack([x_flat, y_flat, z_flat], axis=1)
        return positions
    
    def forward(self, image):
        """
        Simulates RF data from the input image.
        Each pixel is a scatterer, pixel value is magnitude.
        """
        assert len(image.shape)==4, f"Image should be of shape [n_frames, H, W, 1] but got {image.shape}"
        n_frames, *img_shape = image.shape
        magnitudes = ops.reshape(image, (n_frames, -1))[0] #e.g. [17, 12544] --> (12544,)
        element_angles = jnp.zeros(self.scan.n_el)

        rf_data = simulate_rf_data(
            n_ax = self.scan.n_ax,
            scatterer_positions = self.positions[...,[0,2]],
            scatterer_amplitudes = magnitudes,
            t0_delays = self.scan.t0_delays,
            probe_geometry = self.scan.probe_geometry[...,[0,2]],
            element_angles = element_angles,
            tx_apodizations = self.scan.tx_apodizations,
            initial_times = self.scan.initial_times,
            element_width_wl = self.scan.element_width,
            sampling_frequency = self.scan.sampling_frequency,
            carrier_frequency = self.scan.center_frequency,
            sound_speed = self.scan.sound_speed,
        )
        return rf_data
    
    def transpose(self, data):
        raise NotImplementedError("Transpose for SimulateOperatorJaxus is not implemented.")

    def __str__(self):
        return f"SimulateOperator implemented in jax with scan n_tx={self.scan.n_tx}, probe center_freq={self.probe.center_frequency}"

from partial_sim import simulate_partial_rf_data

@operator_registry(name="simulator_partial")
class SimulatorPartial(Operator):
    """
    Operator for simulating RF data from an image.
    The operator is initialized with fixed scan and probe parameters.
    The forward operation takes an image, computes scatterer positions and magnitudes,
    and simulates the RF data using the ultrasound simulator.
    """

    def __init__(self, scan, shape):
        super().__init__()
        self.scan = scan
        self.shape = shape
        self.positions = self.compute_scatterer_positions()

    def compute_scatterer_positions(self):
        """
        Compute scatterer positions for each pixel in the image, using scan x/z limits and image shape.
        Returns an array of shape (n_pixels, 3) with [x, y, z] positions.
        #TODO: Distinguish between scatterers, and from image
        """
        x_start, x_end = self.scan.xlims
        z_start, z_end = self.scan.zlims
        # Accept (H, W), (H, W, C) or (B, H, W, C)
        if len(self.shape) == 2:
            H, W = self.shape
        elif len(self.shape) >= 3:
            H, W = self.shape[-3], self.shape[-2]
        else:
            raise ValueError(f"Invalid shape {self.shape}. Expected (H, W), (H, W, C), or (B, H, W, C).")
        x = ops.linspace(x_start, x_end, H)
        z = ops.linspace(z_start, z_end, W)
        xv, zv = ops.meshgrid(x, z, indexing='ij')
        x_flat = ops.reshape(xv, [-1])
        z_flat = ops.reshape(zv, [-1])
        y_flat = ops.zeros_like(x_flat)
        positions = ops.stack([x_flat, y_flat, z_flat], axis=1)
        return positions
    
    def forward(self, image):
        """
        Simulates RF data from the input image.
        Each pixel is a scatterer, pixel value is magnitude.
        """
        assert len(image.shape)==4, f"Image should be of shape [n_frames, H, W, 1] but got {image.shape}"
        n_frames, *img_shape = image.shape
        magnitudes = ops.reshape(image, (n_frames, -1))[0] #e.g. [17, 12544] --> (12544,)
        element_angles = jnp.zeros(self.scan.n_el)

        ax_indices = jnp.array([512, 700, 40])  # 3 axial indices
        el_indices = jnp.array([50, 80, 90])    # 3 element indices

        rf_data = simulate_partial_rf_data(
            n_ax = self.scan.n_ax,
            scatterer_positions = self.positions[...,[0,2]],
            scatterer_amplitudes = magnitudes,
            t0_delays = self.scan.t0_delays,
            probe_geometry = self.scan.probe_geometry[...,[0,2]],
            element_angles = element_angles,
            tx_apodizations = self.scan.tx_apodizations,
            initial_times = self.scan.initial_times,
            element_width_wl = self.scan.element_width,
            sampling_frequency = self.scan.sampling_frequency,
            carrier_frequency = self.scan.center_frequency,
            ax_indices = ax_indices,
            el_indices = el_indices
        )
        return rf_data
    
    def transpose(self):
        raise NotImplementedError("Transpose for SimulateOperatorJaxus is not implemented.")

    def __str__(self):
        return f"SimulateOperator implemented in jax with scan n_tx={self.scan.n_tx}, probe center_freq={self.probe.center_frequency}"

from infer.simulate import (
    ForwardSettings,
    OptVars,
    inverse_reparameterize_opt_vars,
    inverse_reparameterize_scat_amp,
    inverse_reparameterize_scat_pos,
    reparameterize_opt_vars,
    reparameterize_scat_amp,
    reparameterize_scat_pos,
)

# class SimulatorInfer(Operator):
#     def __init__(self, scan, shape):
#         super().__init__()
#         self.scan = scan.copy()
#         self.shape = shape
#         self.positions = self.get_positions(scan)

#     def compute_scatterer_positions(self):
#         """
#         Compute scatterer positions for each pixel in the image, using scan x/z limits and image shape.
#         Returns an array of shape (n_pixels, 3) with [x, y, z] positions.
#         #TODO: Distinguish between scatterers, and from image
#         """
#         x_start, x_end = self.scan.xlims
#         z_start, z_end = self.scan.zlims
#         # Accept (H, W), (H, W, C) or (B, H, W, C)
#         if len(self.shape) == 2:
#             H, W = self.shape
#         elif len(self.shape) >= 3:
#             H, W = self.shape[-3], self.shape[-2]
#         else:
#             raise ValueError(f"Invalid shape {self.shape}. Expected (H, W), (H, W, C), or (B, H, W, C).")
#         x = ops.linspace(x_start, x_end, H)
#         z = ops.linspace(z_start, z_end, W)
#         xv, zv = ops.meshgrid(x, z, indexing='ij')
#         x_flat = ops.reshape(xv, [-1])
#         z_flat = ops.reshape(zv, [-1])
#         y_flat = ops.zeros_like(x_flat)
#         positions = ops.stack([x_flat, y_flat, z_flat], axis=1)
#         return positions
    
#     def forward(self, image):
#         forward_settings = ForwardSettings(
#             probe_geometry_m=jnp.array(self.scan.probe_geometry),
#             t0_delays_tx_s=jnp.array(self.scan.t0_delays),
#             tgc_gain_curve=jnp.array(self.scan.tgc_gain_curve),            
#             active_element_idx=jnp.array(active_element_idx),
#             initial_times_s=jnp.array(self.scan.initial_times),
#             sound_speed_mps=jnp.array(self.scan.sound_speed),

#             sampling_frequency_hz=jnp.array(self.scan.sampling_frequency),
#             center_frequency_hz=jnp.array(self.scan.center_frequency),
#             tw_indices=jnp.array(
#                 [0], dtype=jnp.int32
#             ),  # TODO # jnp.array(self.scan.tx_waveform_indices, dtype=jnp.int32),
#             element_width_m=jnp.array(self.scan.element_width),
#             pos_vfocus_m=None,  # TODO
#             tx_apodizations_tx=jnp.array(tx_apodizations_tx),
#             waveform_samples=jnp.array(waveform_samples),
#             slope=slope,
#             intercept=intercept,
#             scat_amp_reparameterization=self.scat_amp_reparameterization,
#             n_grad_scat=self.n_grad_scat,
#             forward_model_type=self.forward_model_type,
#             optimize_scatterer_positions=self.optimize_scatterer_positions,
#             apply_lens_correction=self.apply_lens_correction,
#             lens_sound_speed_mps=self.lens_sound_speed_mps,
#             lens_thickness=self.lens_thickness,
#             tx_travel_texture=None,  # TODO
#             rx_travel_texture=None,  # TODO
#             target_region_m=jnp.array(self.target_region_m),
#             symlog_epsilon=self.symlog_epsilon,
#             enable_wavelength_scaling=self.enable_wavelength_scaling,
#         )
        

#     def transpose(self, data, *args, **kwargs):
#         return super().transpose(data, *args, **kwargs)
    
#     def __str__(self):
#         return f"Simulator from INFER"
    
