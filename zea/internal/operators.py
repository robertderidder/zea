"""Measurement operators.

Handles task-dependent operations (A) and noises (n) to simulate a measurement y = Ax + n.

"""

import abc
import zea

import numpy as np
import keras
from keras import ops
import jax
import jax.numpy as jnp

from zea.internal.core import Object
from zea.internal.registry import operator_registry
from zea.func import translate, fori_loop
from zea.simulate_partial import simulate_partial_rf_data
from zea.simulator_jax import simulate_partial_rf_data as sim_jax
from zea.display import scan_convert, compute_scan_convert_2d_coordinates

from zea.simulator_zea_partial import simulate_rf

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

@operator_registry(name="simulator_partial")
class SimulatorPartial(Operator):
    """
    Operator for simulating RF data from an image.
    The operator is initialized with fixed scan parameters.
    The forward operator takes a logcompressed image, normalized between -1 and 1
    It is then translated to dynamic range and log uncompressed and generates rf_data
    """

    def __init__(self, scan, shape, n_ax_samples=10, n_el_samples=10, n_tx_samples=None, wavefront_only=False):
        super().__init__()
        self.scan = scan
        self.shape = shape
        self.positions = self.compute_scatterer_positions()
        self.n_ax_samples = n_ax_samples
        self.n_el_samples = n_el_samples
        self.n_tx_samples = n_tx_samples
        self.wavefront_only = wavefront_only

    def compute_scatterer_positions(self):
        """
        Computes the Cartesian coordinates of every scatterer (pixel) in the
        polar image that the operator is initialised with.

        The input image is assumed to live in ρ–θ space; the rows correspond to
        radial samples between ``scan.rho_range`` and the columns to angles
        between the first and last value of ``scan.polar_angles``.  The
        transformation performed here is identical to the one performed by
        :func:`zea.display.scan_convert_2d` when it is used with the same
        scan parameters; this means that the positions returned here are the
        physical (x,y,z) locations beneath the probe.  Only the x and z
        components are passed to the JAX simulator – the y coordinate is
        zero.
        """
        if self.scan.grid_type == "polar":
            rho_range = (
                self.scan.rho_range,
            )

            theta_range = (
                self.scan.polar_angles[0],
                self.scan.polar_angles[-1],
            )

            # image shape may be (H,W) or (H,W,1) etc.
            assert len(self.shape) == 2, f"Shape must be of form H, W, got {self.shape}"
            H = self.shape[0]
            W = self.shape[1]

            # 1. Create rho-theta grid
            rho = ops.linspace(rho_range[0], rho_range[1], H)
            theta = ops.linspace(theta_range[0], theta_range[1], W)

            #Frustrum rt2xz transformation
            rho_grid, theta_grid = ops.meshgrid(rho, theta, indexing="ij")

            z_grid = rho_grid / ops.sqrt(1 + ops.tan(theta_grid) ** 2)
            x_grid = z_grid * ops.tan(theta_grid)  
            z_grid = z_grid - self.scan.distance_to_apex

            n_pixels = H*W
            x_vec = ops.reshape(x_grid, (n_pixels,))
            y_vec = ops.zeros_like(x_vec)
            z_vec = ops.reshape(z_grid, (n_pixels,))

            positions = ops.stack([x_vec, y_vec, z_vec], axis=1)
        elif self.scan.grid_type == "cartesian":
            """
            Compute scatterer positions for each pixel in the image, using scan x/z limits and image shape.
            Returns an array of shape (n_pixels, 3) with [x, y, z] positions.
            """
            x_start, x_end = self.scan.xlims
            z_start, z_end = self.scan.zlims
            if len(self.shape) == 2:
                H, W = self.shape[0], self.shape[1]
            elif len(self.shape) == 3:
                H, W = self.shape[0], self.shape[1]
            elif len(self.shape) == 4:
                B, H, W = self.shape[0], self.shape[1], self.shape[2]
            else:
                raise ValueError(f"Expected a shape of lenght 2, 3, or 4, instead got {self.shape}") 
            
            x = ops.linspace(x_start, x_end, H)
            z = ops.linspace(z_start, z_end, W)
            xv, zv = ops.meshgrid(x, z, indexing='ij')
            x_flat = ops.reshape(xv, [-1])
            z_flat = ops.reshape(zv, [-1])
            y_flat = ops.zeros_like(x_flat)
            positions = ops.stack([x_flat, y_flat, z_flat], axis=1)
        else:
            raise ValueError(f"Invalid grid type {self.scan.grid_type}. Expected 'polar' or 'cartesian'.")
        return positions
    
    def img_to_magnitude(self,image):
        image = translate(image, range_from = (-1,1), range_to=self.scan.dynamic_range)
        image_lin = 10**(image/20)
        return image_lin

    def forward(self, image, seed=None, **kwargs):
        """
        Simulates RF data from the input images.
        Image magnitudes have values between -1 and 1
        Each pixel is a scatterer, pixel value is magnitude.
        """
        assert len(image.shape)==4, f"Image should be of shape [n_frames, H, W, 1] but got {image.shape}"
        if seed is None:
            raise ValueError("A random seed must be provided.")
        n_frames, *img_shape = image.shape
        magnitudes = self.img_to_magnitude(image)
        magnitudes = ops.reshape(magnitudes, (n_frames, -1))

        ax_indices = jax.random.choice(seed,self.scan.n_ax,(self.n_ax_samples,),replace=False)
        el_indices = jax.random.choice(seed,self.scan.n_el,(self.n_el_samples,),replace=False)
        tx_indices = jax.random.choice(seed,self.scan.n_tx,(self.n_tx_samples,),replace=False)

        ax_indices = jax.numpy.sort(ax_indices)
        el_indices = jax.numpy.sort(el_indices)
        tx_indices = jax.numpy.sort(tx_indices)

        rf_data = sim_jax(
            ax_indices = ax_indices,
            el_indices = el_indices,
            tx_indices = tx_indices,
            scatterer_positions = self.positions[...,[0,2]],
            scatterer_amplitudes = magnitudes,
            t0_delays = jnp.array(self.scan.t0_delays),
            probe_geometry = jnp.array(self.scan.probe_geometry[...,[0,2]]),
            element_angles = jnp.zeros(self.scan.n_el),
            tx_apodizations = jnp.array(self.scan.tx_apodizations),
            initial_times = jnp.array(self.scan.initial_times),
            element_width_wl = jnp.array(self.scan.element_width/self.scan.wavelength),
            carrier_frequency = self.scan.center_frequency,
            sampling_frequency = self.scan.sampling_frequency,
            wavefront_only=self.wavefront_only,
            waveform_samples = None,
            **kwargs
        )
        return (rf_data, tx_indices, ax_indices, el_indices)
    
    def transpose(self):
        raise NotImplementedError("Transpose for SimulateOperatorJaxus is not implemented.")

    def __str__(self):
        return f"SimulateOperator implemented in jax"


class SimulatorPartialFFT(Operator):
    def __init__(self, scan, shape, n_el_samples = 10, n_freq_samples = 10, n_tx_samples = 5, wavefront_only=False, scatterer_chunk_size = 32):
        super().__init__()
        self.scan = scan
        self.shape = shape
        self.n_el_samples = n_el_samples
        self.n_freqs = 2**(jnp.ceil(jnp.log2(self.scan.n_ax))) // 2 + 1
        self.n_freq_samples = n_freq_samples
        self.n_tx_samples = n_tx_samples
        self.wavefront_only = wavefront_only
        self.positions = self.compute_scatterer_positions()
        self.scatterer_chunk_size = scatterer_chunk_size

    def fit_point_scatterers(image):
        """
        does this make sens? sampling at every iteration seems wasetful. initializing once, and updating along with amplitudes seems better.
        In that case I would only have to sample from the initial guess which is the tweedy estimate
        https://github.com/tue-bmd/Bayesian-REFoCUS/blob/echonetlvh/simulation/fit_point_scatterers.py
        """
        return 0
    
    def img_to_magnitude(self,image):
        image = translate(image, range_from = (-1,1), range_to=self.scan.dynamic_range)
        image_lin = 10**(image/20)
        return image_lin
    
    def compute_scatterer_positions(self):
        """
        Computes the Cartesian coordinates of every scatterer (pixel) in the
        polar image that the operator is initialised with.

        The input image is assumed to live in ρ–θ space; the rows correspond to
        radial samples between ``scan.rho_range`` and the columns to angles
        between the first and last value of ``scan.polar_angles``.  The
        transformation performed here is identical to the one performed by
        :func:`zea.display.scan_convert_2d` when it is used with the same
        scan parameters; this means that the positions returned here are the
        physical (x,y,z) locations beneath the probe.  Only the x and z
        components are passed to the JAX simulator – the y coordinate is
        zero.
        """
        if self.scan.grid_type == "polar":
            rho_range = (
                self.scan.rho_range,
            )

            theta_range = (
                self.scan.polar_angles[0],
                self.scan.polar_angles[-1],
            )

            # image shape may be (H,W) or (H,W,1) etc.
            assert len(self.shape) == 2, f"Shape must be of form H, W, got {self.shape}"
            H = self.shape[0]
            W = self.shape[1]

            # 1. Create rho-theta grid
            rho = ops.linspace(rho_range[0], rho_range[1], H)
            theta = ops.linspace(theta_range[0], theta_range[1], W)

            #Frustrum rt2xz transformation
            rho_grid, theta_grid = ops.meshgrid(rho, theta, indexing="ij")

            z_grid = rho_grid / ops.sqrt(1 + ops.tan(theta_grid) ** 2)
            x_grid = z_grid * ops.tan(theta_grid)  
            z_grid = z_grid - self.scan.distance_to_apex

            n_pixels = H*W
            x_vec = ops.reshape(x_grid, (n_pixels,))
            y_vec = ops.zeros_like(x_vec)
            z_vec = ops.reshape(z_grid, (n_pixels,))

            positions = ops.stack([x_vec, y_vec, z_vec], axis=1)
        elif self.scan.grid_type == "cartesian":
            """
            Compute scatterer positions for each pixel in the image, using scan x/z limits and image shape.
            Returns an array of shape (n_pixels, 3) with [x, y, z] positions.
            """
            x_start, x_end = self.scan.xlims
            z_start, z_end = self.scan.zlims
            if len(self.shape) == 2:
                H, W = self.shape[0], self.shape[1]
            elif len(self.shape) == 3:
                H, W = self.shape[0], self.shape[1]
            elif len(self.shape) == 4:
                B, H, W = self.shape[0], self.shape[1], self.shape[2]
            else:
                raise ValueError(f"Expected a shape of lenght 2, 3, or 4, instead got {self.shape}") 
            
            x = ops.linspace(x_start, x_end, H)
            z = ops.linspace(z_start, z_end, W)
            xv, zv = ops.meshgrid(x, z, indexing='ij')
            x_flat = ops.reshape(xv, [-1])
            z_flat = ops.reshape(zv, [-1])
            y_flat = ops.zeros_like(x_flat)
            positions = ops.stack([x_flat, y_flat, z_flat], axis=1)
        else:
            raise ValueError(f"Invalid grid type {self.scan.grid_type}. Expected 'polar' or 'cartesian'.")
        return positions
    
    def forward(self,image, seed):
        n_frames, *img_shape = image.shape
        magnitudes = self.img_to_magnitude(image)
        magnitudes = ops.reshape(magnitudes, (n_frames, -1))

        seed, el_seed, tx_seed, freq_seed = jax.random.split(seed, 4)
        
        el_indices = jax.random.choice(el_seed, self.scan.n_el, (self.n_el_samples,), replace=False)
        tx_indices = jax.random.choice(tx_seed, self.scan.n_tx, (self.n_tx_samples,), replace=False)

        # Sample frequencies with Gaussian weighting centered at carrier frequency
        freq_bins = jnp.arange(self.n_freqs, dtype=jnp.float32)
        std_bins = self.n_freqs // 8
        probs = jnp.exp(-0.5 * ((freq_bins - self.n_freqs/2) / std_bins) ** 2)
        probs = probs / probs.sum()
        
        freq_indices = jax.random.choice(freq_seed, self.n_freqs, (self.n_freq_samples,),replace=False)
        
        el_indices = jax.numpy.sort(el_indices)
        freq_indices = jax.numpy.sort(freq_indices)
        tx_indices = jax.numpy.sort(tx_indices)
        
        # Compute scatterer padding for chunking
        n_scat = self.positions.shape[0]
        chunk_size = self.scatterer_chunk_size
        n_scat_padded = n_scat + (chunk_size - (n_scat % chunk_size)) % chunk_size
        
        # Pad positions and use frame loop
        positions_padded = ops.pad(
            self.positions, 
            ((0, n_scat_padded - n_scat), (0, 0))
        )
        
        def process_frame(frame_idx, rf_data_all):
            """Process one frame by looping over scatterer chunks."""
            amplitudes = magnitudes[frame_idx]
            
            # Pad amplitudes for chunking
            amplitudes_padded = ops.pad(
                amplitudes,
                (0, n_scat_padded - n_scat)
            )
            
            # Initialize accumulated RF for this frame
            rf_accum = ops.zeros([self.n_tx_samples, self.n_freq_samples, self.n_el_samples, 1], dtype='complex64')
            
            def process_scat_chunk(chunk_idx, rf_acc):
                """Process one scatterer chunk and accumulate."""
                start = chunk_idx * chunk_size
                
                scat_chunk = ops.slice(positions_padded, (start, 0), (chunk_size, 3))
                amp_chunk = ops.slice(amplitudes_padded, (start,), (chunk_size,))
                
                rf_chunk = simulate_rf(
                        scatterer_positions=scat_chunk,
                        scatterer_magnitudes=amp_chunk,
                        el_indices=el_indices,
                        freq_indices=freq_indices,
                        tx_indices=tx_indices,
                        probe_geometry=self.scan.probe_geometry,
                        apply_lens_correction=self.scan.apply_lens_correction,
                        sound_speed=self.scan.sound_speed,
                        lens_sound_speed=self.scan.lens_sound_speed,
                        lens_thickness=self.scan.lens_thickness,
                        n_ax=self.scan.n_ax,
                        center_frequency=self.scan.center_frequency,
                        sampling_frequency=self.scan.sampling_frequency,
                        t0_delays=self.scan.t0_delays,
                        initial_times=self.scan.initial_times,
                        element_width=self.scan.element_width,
                        attenuation_coef=self.scan.attenuation_coef,
                        tx_apodizations=self.scan.tx_apodizations,
                )
                
                # RF is linear in amplitudes, so sum contributions
                return rf_acc + rf_chunk
            
            # Loop over scatterer chunks and accumulate
            n_chunks = n_scat_padded // chunk_size
            rf_accum = fori_loop(0, n_chunks, process_scat_chunk, rf_accum)[None]
            
            # Write accumulated RF to output
            rf_data_all = ops.slice_update(rf_data_all, (frame_idx, 0, 0, 0, 0), rf_accum)
            return rf_data_all
        
        # Pre-allocate output: (n_frames, n_tx_sel, n_el_sel, n_freq_sel, 1)
        output_shape = (n_frames, self.n_tx_samples, self.n_freq_samples, self.n_el_samples, 1)
        rf_data_all = ops.zeros(output_shape, dtype='complex64')

        # Process all frames with fori_loop
        rf_data = fori_loop(0, n_frames, process_frame, rf_data_all)
        
        return rf_data,  tx_indices, freq_indices, el_indices
    
    def transpose(self):
        raise NotImplementedError("Transpose for SimulateOperatorJaxus is not implemented.")

    def __str__(self):
        return f"SimulateOperator implemented in jax"

