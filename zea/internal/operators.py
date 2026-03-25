"""Measurement operators.

Handles task-dependent operations (A) and noises (n) to simulate a measurement y = Ax + n.

"""

import abc
import zea

import numpy as np
import keras
from keras import ops
import jax

from zea.internal.core import Object
from zea.internal.registry import operator_registry
from zea.func import translate
from zea.func.tensor import split_seed
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

class Simulator(Operator):
    def __init__(self, scan, 
                 n_tx_samples = 5, 
                 n_freq_samples = 10, 
                 n_el_samples = 10, 
                 scatterer_chunk_size = 256, 
                 n_scat_per_it = None):
        super().__init__()
        self.scan = scan
        self.shape = self.scan.grid.shape[:2]
        self.n_el_samples = n_el_samples
        self.n_freqs = 2**int(ops.ceil(ops.log2(self.scan.n_ax))) // 2 + 1
        self.n_freq_samples = n_freq_samples
        self.n_tx_samples = n_tx_samples
        self.positions = ops.reshape(self.scan.grid, (-1, 3))
        self.scatterer_chunk_size = scatterer_chunk_size
        if n_scat_per_it is None:
            self.n_scat_per_it = int(self.scan.grid.shape[0] * self.scan.grid.shape[1])
        else:
            self.n_scat_per_it = int(n_scat_per_it)
    
    def img_to_magnitude(self,image, n_frames):
        image = translate(image, range_from = (-1,1), range_to=self.scan.dynamic_range)
        image_lin = 10**(image/20)
        image_lin = ops.reshape(image_lin, (n_frames, -1))
        #set amplitudes to 0 if scan.grid has z<0
        mask = ops.logical_not(self.positions[:,2] < 0)[None,:]
        image_lin = mask * image_lin
        return image_lin
    
    def sample_indices(self, seed):
        n_scat = len(self.positions)
        # Uniform sampling without replacement using random sort trick
        random_vals = keras.random.uniform(shape=(n_scat,), seed=seed)
        indices = ops.argsort(random_vals)[:self.n_scat_per_it]
        indices = ops.sort(indices)
        return indices
    
    def forward(self,image, seed, **kwargs):
        assert len(image.shape)==4, f"Image should be of shape [n_frames, H, W, 1] but got {image.shape}"
        image = ops.image.resize(image, (self.scan.grid_size_x, self.scan.grid_size_z))
        scat_seed, el_seed, tx_seed, freq_seed = split_seed(seed, 4)   

        n_frames, *img_shape = image.shape
        magnitudes = self.img_to_magnitude(image, n_frames)

        scat_indices = self.sample_indices(scat_seed)

        positions = ops.take(self.positions, scat_indices, axis=0)
        magnitudes = ops.take(magnitudes, scat_indices, axis=1)

        # Uniform sampling without replacement for elements and transmits
        el_random = keras.random.uniform(shape=(self.scan.n_el,), seed=el_seed)
        el_indices = ops.argsort(el_random)[:self.n_el_samples]
        el_indices = ops.sort(el_indices)
        
        tx_random = keras.random.uniform(shape=(self.scan.n_tx,), seed=tx_seed)
        tx_indices = ops.argsort(tx_random)[:self.n_tx_samples]
        tx_indices = ops.sort(tx_indices)

        # Sample frequencies with Gaussian weighting centered at carrier frequency
        freq_bins = ops.arange(self.n_freqs, dtype='float32')
        std_bins = self.n_freqs // 8
        probs = ops.exp(-0.5 * ((freq_bins - self.n_freqs/2) / std_bins) ** 2)
        probs = probs / probs.sum()
        
        #Apply Gumbel-max trick for sampling without replacement according to probs
        log_probs = ops.log(ops.maximum(probs, 1e-10))  # Avoid log(0)
        gumbel = -ops.log(-ops.log(keras.random.uniform(shape=(self.n_freqs,), seed=freq_seed)) + 1e-10)
        freq_indices = ops.argsort(-(log_probs + gumbel))[:self.n_freq_samples]
        freq_indices = ops.sort(freq_indices)
        # freq_indices = jax.random.choice(freq_seed, self.n_freqs, (self.n_freq_samples,),replace=False, p=probs)
        
        el_indices = ops.sort(el_indices)
        freq_indices = ops.sort(freq_indices)
        tx_indices = ops.sort(tx_indices)
        
        # Compute scatterer padding for chunking
        n_scat = positions.shape[0]
        chunk_size = self.scatterer_chunk_size
        n_scat_padded = n_scat + (chunk_size - (n_scat % chunk_size)) % chunk_size
        n_chunks = n_scat_padded // chunk_size

        # Pad and pre-reshape positions into (n_chunks, chunk_size, 3)
        positions_padded = ops.pad(positions, ((0, n_scat_padded - n_scat), (0, 0)))
        scat_chunks = ops.reshape(positions_padded, (n_chunks, chunk_size, 3))

        def simulate_chunk(scat_chunk, amp_chunk):
            """Simulate RF for a single scatterer chunk."""
            return simulate_rf(
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

        # Checkpoint the scan body functions so JAX recomputes their internals
        # during the backward pass rather than storing all intermediate carry states.
        # - accumulate_chunks: avoids storing n_chunks copies of rf_acc
        # - process_frame: avoids storing n_frames (= batch_size * n_samples) copies of
        #   intermediate activations — this is the dominant term that grows with n_samples.
        @jax.checkpoint
        def accumulate_chunks(rf_acc, chunk):
            """Accumulate RF contribution from one scatterer chunk into carry."""
            scat_chunk, amp_chunk = chunk
            rf_acc = rf_acc + simulate_chunk(scat_chunk, amp_chunk)
            return rf_acc, None

        @jax.checkpoint
        def process_frame(carry, frame_magnitudes):
            """Scan over scatterer chunks for a single frame; output is the accumulated RF."""
            amplitudes_padded = ops.pad(frame_magnitudes, (0, n_scat_padded - n_scat))
            amp_chunks = ops.reshape(amplitudes_padded, (n_chunks, chunk_size))

            rf_init = ops.zeros(
                [self.n_tx_samples, self.n_freq_samples, self.n_el_samples, 1], dtype='complex64'
            )
            # Inner scan: accumulate RF over scatterer chunks (carry), no per-chunk output needed
            rf_accum, _ = ops.scan(accumulate_chunks, rf_init, (scat_chunks, amp_chunks))
            return carry, rf_accum

        # Outer scan: xs = magnitudes (n_frames, n_scat); outputs stacked -> (n_frames, ...)
        _, rf_data = ops.scan(process_frame, None, magnitudes)
        
        return 10000*rf_data, tx_indices, freq_indices, el_indices
    
    def transpose(self):
        raise NotImplementedError("The transpose of the simulator operator is not implemented.")

    def __str__(self):
        return f"y = A*x where A is simulator in the fourier domain. Only returns a subset of the indices."


class Simulator_Experimental(Operator):
    """
    Experimental version of simulator operator. Extra features include:
        - Option to add jitter to scatterer positions to break symmetries and make the problem more realistic. Choose from 0 (no jitter), 1 (jitter once at start) or 2 (jitter at each iteration). 
        - Option to sample scatterers based on intensity of the input image, rather than uniformly at random. 
          This can be used to simulate a more realistic scenario where scatterer points are in in high-intensity regions
    """
    def __init__(self, scan, 
                 n_tx_samples = 5, 
                 n_freq_samples = 10, 
                 n_el_samples = 10, 
                 scatterer_chunk_size = 256, 
                 n_scat_per_it = None,
                 add_jitter = 0,
                 sample_by_intensity = False,
                 jitter_scale = 0.1,):
        super().__init__()
        self.scan = scan
        self.shape = self.scan.grid.shape[:2]
        self.n_el_samples = n_el_samples
        self.n_freqs = 2**int(ops.ceil(ops.log2(self.scan.n_ax))) // 2 + 1
        self.n_freq_samples = n_freq_samples
        self.n_tx_samples = n_tx_samples
        self.positions = self.scan.flatgrid
        self.add_jitter = add_jitter
        self.sample_by_intensity = sample_by_intensity
        self.scatterer_chunk_size = scatterer_chunk_size
        self.extent = self.scan.extent
        self.jitter_scale = jitter_scale
        if n_scat_per_it is None:
            self.n_scat_per_it = int(self.scan.grid.shape[0] * self.scan.grid.shape[1])
        else:
            self.n_scat_per_it = int(n_scat_per_it)

        if self.add_jitter == 1:
            # Add jitter once at the start
            dx = (self.positions[...,0].max() - self.positions[...,0].min())/self.shape[0]
            dz = (self.positions[...,2].max() - self.positions[...,2].min())/self.shape[1]
            
            jitter_scale_x = self.jitter_scale*dx  # Adjust as needed
            jitter_scale_z = self.jitter_scale*dz  # Adjust as needed
            jit_x_seed, jit_z_seed = split_seed(jax.random.key(0), 2)
            jitter_x = keras.random.normal(shape=(self.positions.shape[0],), seed=jit_x_seed) * jitter_scale_x
            jitter_z = keras.random.normal(shape=(self.positions.shape[0],), seed=jit_z_seed) * jitter_scale_z
            jitter = ops.stack([jitter_x, ops.zeros_like(jitter_x), jitter_z], axis=1)
            self.positions = self.positions + jitter
    
    
    def img_to_magnitude(self,image, n_frames):
        image = translate(image, range_from = (-1,1), range_to=self.scan.dynamic_range)
        image_lin = 10**(image/20)
        image_lin = ops.reshape(image_lin, (n_frames, -1))
        #set amplitudes to 0 if scan.grid has z<0
        mask = ops.logical_not(self.positions[:,2] < 0)[None,:]
        image_lin = mask * image_lin
        return image_lin
    
    def sample_indices_by_intensity(self, magnitudes, seed, progress):
        #probability distribution is uniform at progress = 0, and becomes more and more weighted towards high intensity regions as progress approaches 1.
        n_scat = len(self.positions)
        intensity = ops.mean(magnitudes, axis=0) # (n_scat,)
        probs = intensity / ops.sum(intensity)

        #probs based on intensity:
        probs_intensity = ops.nan_to_num(probs, nan=1.0/n_scat)
        #uniform probs
        probs_uniform = ops.ones_like(probs_intensity) / n_scat

        #Interpolate between uniform and intensity-based sampling based on progress
        #TODO: test if a sinuoidal schedule for progress improves results compared to linear
        probs = (1-progress)*probs_uniform + progress*probs_intensity

        #Apply Gumbel-max trick for sampling without replacement according to probs
        log_probs = ops.log(ops.maximum(probs, 1e-10))  # Avoid log(0)
        gumbel = -ops.log(-ops.log(keras.random.uniform(shape=(n_scat,), seed=seed)) + 1e-10)
        indices = ops.argsort(-(log_probs + gumbel))[:self.n_scat_per_it]
        indices = ops.sort(indices)
        return indices

    def sample_indices(self, magnitudes, seed, sample_by_intensity, **kwargs):
        #progress is float between 0 and 1, indicating how far long the diffusion process we are.
        n_scat = len(self.positions)
        if not sample_by_intensity: # Uniform sampling without replacement using random sort trick
            random_vals = keras.random.uniform(shape=(n_scat,), seed=seed)
            indices = ops.argsort(random_vals)[:self.n_scat_per_it]
            indices = ops.sort(indices)
        if sample_by_intensity: # Sample based on intensity of input image, more weighting as progress goes toward 1.
            indices = self.sample_indices_by_intensity(magnitudes, seed, **kwargs)
        return indices
    
    def forward(self, image, seed, **kwargs):
        assert len(image.shape)==4, f"Image should be of shape [n_frames, H, W, 1] but got {image.shape}"
        image = ops.image.resize(image, self.shape)
        scat_seed, el_seed, tx_seed, freq_seed, jit_x_seed, jit_z_seed = split_seed(seed, 6)   

        n_frames, *img_shape = image.shape
        magnitudes = self.img_to_magnitude(image, n_frames)

        scat_indices = self.sample_indices(magnitudes, scat_seed, sample_by_intensity=self.sample_by_intensity, **kwargs)

        positions = ops.take(self.positions, scat_indices, axis=0)
        magnitudes = ops.take(magnitudes, scat_indices, axis=1)

        if self.add_jitter == 2: # Add jitter at each iteration
            dx = (self.positions[...,0].max() - self.positions[...,0].min())/self.shape[0]
            dz = (self.positions[...,2].max() - self.positions[...,2].min())/self.shape[1]
            
            # Add small jitter to positions to break symmetries and make the problem more realistic
            jitter_scale_x = self.jitter_scale*dx  # Adjust as needed
            jitter_scale_z = self.jitter_scale*dz  # Adjust as needed
            jitter_x = keras.random.normal(shape=(positions.shape[0],), seed=jit_x_seed) * jitter_scale_x
            jitter_z = keras.random.normal(shape=(positions.shape[0],), seed=jit_z_seed) * jitter_scale_z
            jitter = ops.stack([jitter_x, ops.zeros_like(jitter_x), jitter_z], axis=1)
            positions = positions + jitter

        # Uniform sampling without replacement for elements and transmits
        el_random = keras.random.uniform(shape=(self.scan.n_el,), seed=el_seed)
        el_indices = ops.argsort(el_random)[:self.n_el_samples]
        el_indices = ops.sort(el_indices)
        
        tx_random = keras.random.uniform(shape=(self.scan.n_tx,), seed=tx_seed)
        tx_indices = ops.argsort(tx_random)[:self.n_tx_samples]
        tx_indices = ops.sort(tx_indices)

        # Sample frequencies with Gaussian weighting centered at carrier frequency
        freq_bins = ops.arange(self.n_freqs, dtype='float32')
        std_bins = self.n_freqs // 8
        probs = ops.exp(-0.5 * ((freq_bins - self.n_freqs/2) / std_bins) ** 2)
        probs = probs / probs.sum()
        
        #Apply Gumbel-max trick for sampling without replacement according to probs
        log_probs = ops.log(ops.maximum(probs, 1e-10))  # Avoid log(0)
        gumbel = -ops.log(-ops.log(keras.random.uniform(shape=(self.n_freqs,), seed=freq_seed)) + 1e-10)
        freq_indices = ops.argsort(-(log_probs + gumbel))[:self.n_freq_samples]
        freq_indices = ops.sort(freq_indices)
        
        el_indices = ops.sort(el_indices)
        freq_indices = ops.sort(freq_indices)
        tx_indices = ops.sort(tx_indices)
        
        # Compute scatterer padding for chunking
        n_scat = positions.shape[0]
        chunk_size = self.scatterer_chunk_size
        n_scat_padded = n_scat + (chunk_size - (n_scat % chunk_size)) % chunk_size
        n_chunks = n_scat_padded // chunk_size

        # Pad and pre-reshape positions into (n_chunks, chunk_size, 3)
        positions_padded = ops.pad(positions, ((0, n_scat_padded - n_scat), (0, 0)))
        scat_chunks = ops.reshape(positions_padded, (n_chunks, chunk_size, 3))

        def simulate_chunk(scat_chunk, amp_chunk):
            """Simulate RF for a single scatterer chunk."""
            return simulate_rf(
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

        # Checkpoint the scan body functions so JAX recomputes their internals
        # during the backward pass rather than storing all intermediate carry states.
        # - accumulate_chunks: avoids storing n_chunks copies of rf_acc
        # - process_frame: avoids storing n_frames (= batch_size * n_samples) copies of
        #   intermediate activations — this is the dominant term that grows with n_samples.
        @jax.checkpoint
        def accumulate_chunks(rf_acc, chunk):
            """Accumulate RF contribution from one scatterer chunk into carry."""
            scat_chunk, amp_chunk = chunk
            rf_acc = rf_acc + simulate_chunk(scat_chunk, amp_chunk)
            return rf_acc, None

        @jax.checkpoint
        def process_frame(carry, frame_magnitudes):
            """Scan over scatterer chunks for a single frame; output is the accumulated RF."""
            amplitudes_padded = ops.pad(frame_magnitudes, (0, n_scat_padded - n_scat))
            amp_chunks = ops.reshape(amplitudes_padded, (n_chunks, chunk_size))

            rf_init = ops.zeros(
                [self.n_tx_samples, self.n_freq_samples, self.n_el_samples, 1], dtype='complex64'
            )
            # Inner scan: accumulate RF over scatterer chunks (carry), no per-chunk output needed
            rf_accum, _ = ops.scan(accumulate_chunks, rf_init, (scat_chunks, amp_chunks))
            return carry, rf_accum

        # Outer scan: xs = magnitudes (n_frames, n_scat); outputs stacked -> (n_frames, ...)
        _, rf_data = ops.scan(process_frame, None, magnitudes)
        
        return 10000*rf_data, tx_indices, freq_indices, el_indices
    
    def transpose(self):
        raise NotImplementedError("The transpose of the simulator operator is not implemented.")

    def __str__(self):
        return f"y = A*x where A is simulator in the fourier domain. Only returns a subset of the indices."
