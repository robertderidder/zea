"""Measurement operators.

Handles task-dependent operations (A) and noises (n) to simulate a measurement y = Ax + n.

"""

import abc

import numpy as np
from keras import ops

from zea.internal.core import Object
from zea.internal.registry import operator_registry

#own imports
from zea.func.tensor import split_seed, translate
from zea import log
import keras
from zea.simulator_subsample import simulate_rf
import jax
import jax.numpy as jnp

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
    """
    Simulator opeator class. This simulates rf data given an image. 
    The simulation is done by converting each pixel into a scatter point
    """
    def __init__(self,
                 scan,
                 n_tx_samples,
                 n_freq_samples,
                 n_el_samples,
                 scatterer_chunk_size = 256,):
        super().__init__()
        self.scan = scan
        self.n_tx_samples = n_tx_samples
        self.n_freq_samples = n_freq_samples
        self.n_el_samples = n_el_samples
        self.scatterer_chunk_size = scatterer_chunk_size
        self.positions = self.scan.flatgrid
        self.n_freqs = 2**int(ops.ceil(ops.log2(self.scan.n_ax))) // 2 + 1

    def forward(self, image, seed, freq_gaussian_probs=False):
        assert len(image.shape) == 4, "Image should have shape (B, H, W, C)"
        seed_el, seed_freq, seed_tx = split_seed(seed, 3)
        
        n_frames, *img_shape = image.shape
        magnitudes = self.image_to_magnitudes(image, n_frames)

        #sample el, freq and tx
        el_random = keras.random.uniform(shape=(self.scan.n_el,), seed=seed_el)
        el_indices = ops.argsort(el_random)[:self.n_el_samples]

        def _gaussian_probs(_):
            freq_bins = ops.cast(ops.arange(self.n_freqs), "float32")
            center_freq_bin = ops.cast(self.n_freqs // 2, "float32")
            sigma = ops.cast(self.n_freqs / 8, "float32")
            gaussian_probs = ops.exp(-0.5 * ((freq_bins - center_freq_bin) / sigma) ** 2)
            return gaussian_probs / ops.sum(gaussian_probs)

        def _uniform_probs(_):
            return ops.ones((self.n_freqs,), dtype="float32") / self.n_freqs

        p = jax.lax.cond(ops.cast(freq_gaussian_probs, "bool"), _gaussian_probs, _uniform_probs, None)

        log_probs = ops.log(ops.maximum(p, 1e-10))
        gumbel = -ops.log(-ops.log(keras.random.uniform(shape=(self.n_freqs,), seed=seed_freq)))
        freq_random = log_probs + gumbel
        freq_indices = ops.argsort(freq_random)[:self.n_freq_samples]

        tx_random = keras.random.uniform(shape=(self.scan.n_tx,), seed=seed_tx)
        tx_indices = ops.argsort(tx_random)[:self.n_tx_samples]

        el_indices = ops.sort(el_indices)
        freq_indices = ops.sort(freq_indices)
        tx_indices = ops.sort(tx_indices)

        # Compute scatterer padding for chunking
        n_scat = self.positions.shape[0]
        chunk_size = self.scatterer_chunk_size
        n_scat_padded = n_scat + (chunk_size - (n_scat % chunk_size)) % chunk_size
        n_chunks = n_scat_padded // chunk_size

        # Pad and pre-reshape positions into (n_chunks, chunk_size, 3)
        positions_padded = ops.pad(self.positions, ((0, n_scat_padded - n_scat), (0, 0)))
        position_chunks = ops.reshape(positions_padded, (n_chunks, chunk_size, 3))
        magnitudes_padded = ops.pad(magnitudes, ((0, 0), (0, n_scat_padded - n_scat)))
        
        def simulate_chunk(pos_chunk, mag_chunk):
            """Simulate RF for a single scatterer chunk."""
            return simulate_rf(
                scatterer_positions=pos_chunk,
                scatterer_magnitudes=mag_chunk,
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
        
        @jax.checkpoint
        def accumulate_chunks(rf_accumulated, chunk):
            pos_chunk, mag_chunk = chunk
            rf_accumulated += simulate_chunk(pos_chunk, mag_chunk)
            return rf_accumulated, None

        @jax.checkpoint
        def process_frame(carry, frame_data):
            magnitude_chunks = ops.reshape(frame_data, (n_chunks, chunk_size))
            rf_init = ops.zeros((self.n_tx_samples, self.n_freq_samples, self.n_el_samples, 1), dtype='complex64')
            rf_accumulated, _ = ops.scan(accumulate_chunks, rf_init, (position_chunks, magnitude_chunks))
            return carry, rf_accumulated
        
        _, rf_data = ops.scan(process_frame, None, magnitudes_padded)

        return rf_data, tx_indices, freq_indices, el_indices

    def image_to_magnitudes(self, image, n_frames):
        image = translate(image, range_from=(-1,1), range_to=self.scan.dynamic_range)
        image_lin = 10**(image/20)  # convert from dB to linear scale
        image_lin = ops.reshape(image_lin, (n_frames, -1))
        #set magnitudes of pixels z<0 to 0r
        mask = ops.logical_not(self.positions[:,2] < 0)[None, :]
        image_lin = image_lin * mask
        return image_lin

    def __str__(self):
        return f"Simulator with {self.n_tx_samples} tx samples, {self.n_freq_samples} freq samples, and {self.n_el_samples} el samples. Not optimizing auxiliary variables."

    def transpose(self, data, *args, **kwargs):
        raise NotImplementedError("Transpose not implemented for Simulator operator.")
    
class Simulator_Total(Operator):
    """ Simulator that optimizes other auxiliary variables. 
    This may not depend on the dynamic range, because that is a visualization parameter, not part of the forward model """

    def __init__(self,
             scan,
             n_tx_samples,
             n_freq_samples,
             n_el_samples,
             scatterer_chunk_size = 256,
             position_offset_wl = 0.0,
             sound_speed_offset_scale = 100.0,
             n_equidistant_beams = 0,
             n_random_beams = 0,):
        super().__init__()
        self.scan = self.validate_scan(scan)
        self.shape = scan.grid.shape[:2]
        self.n_tx_samples = n_tx_samples
        self.n_freqs = 2**int(ops.ceil(ops.log2(self.scan.n_ax))) // 2 + 1
        self.n_freq_samples = n_freq_samples
        self.n_el_samples = n_el_samples
        self.scatterer_chunk_size = scatterer_chunk_size
        self.sound_speed_offset_scale = sound_speed_offset_scale
        self.positions = scan.flatgrid + 2 * position_offset_wl * (keras.random.uniform(shape=scan.flatgrid.shape, seed=42)-ops.ones_like(self.scan.flatgrid)*0.5)* scan.wavelength
        self.beams = self.select_beams(n_equidistant_beams, n_random_beams)

    def validate_scan(self, scan):
        required_attrs = ["probe_geometry","apply_lens_correction","sound_speed","lens_sound_speed",
                          "lens_thickness","n_ax","center_frequency","sampling_frequency","t0_delays",
                          "initial_times","element_width","attenuation_coef","tx_apodizations"]
    
        if not hasattr(scan, "apply_lens_correction"):
            scan.apply_lens_correction = False  # Default to False if not provided
            scan.lens_sound_speed = scan.sound_speed  # Default to same as sound speed
            scan.lens_thickness = 0.0  # Default to no lens thickness
            log.warning("Scan object missing lens correction attributes. Defaulting to no lens correction.")

        for attr in required_attrs:
            if not hasattr(scan, attr):
                raise ValueError(f"Scan object is missing required attribute: {attr}")
        return scan
    
    def select_beams(self, n_equidistant, n_random):
        if n_equidistant == "all" or n_random == "all":
            return ops.arange(self.scan.n_tx, dtype="int32")
        
        if n_equidistant == 0 and n_random == 0:
            raise ValueError("At least one of n_equidistant_beams or n_random_beams must be greater than 0.")
        if n_equidistant > 0 and n_random > 0:
            raise ValueError("n_equidistant_beams and n_random_beams cannot both be greater than 0. Please choose one sampling strategy.")
        if n_equidistant > 0:
            #choose n equidistant transmits from the scan
            beams = ops.linspace(0, self.scan.n_tx-1, n_equidistant, dtype="int32")
        if n_random > 0:
            #choose n random transmits from the scan
            beams = ops.random.shuffle(ops.arange(self.scan.n_tx, dtype="int32"))[:n_random]
        if self.n_tx_samples > len(beams):
            raise ValueError(f"n_tx_samples ({self.n_tx_samples}) cannot be greater than the number of selected beams ({len(beams)}). Please adjust n_tx_samples or the beam selection strategy.")
        return beams

    def img_to_magnitude(self, image, positions):
        n_frames = image.shape[0]
        image = ops.reshape(image, (n_frames, -1))
        image = self.symexp(image)
        mask = ops.logical_not(positions[:,2] < 0)[None, :]
        image = image * mask
        return image
    
    def symlog(self, x, epsilon=0.01):
        x_scaled = x / epsilon
        return ops.sign(x) * ops.log1p(ops.abs(x_scaled))
    
    def symexp(self, x, epsilon=0.01):
        a = epsilon * ops.sign(x) * (ops.exp(x) - 1)
        return a
    
    def reparameterize_optvars(self, optvars):
        """Reparameterize optimization variables.
        
        Only unpack and scale variables actually present in optvars to avoid
        unnecessary tracing through unused array creations during autograd.
        """
        out = dict(optvars)
        
        # Sound speed: only recompute if present
        if "sound_speed_offset" in out:
            sound_speed_offset = out["sound_speed_offset"] * self.sound_speed_offset_scale
            sound_speed = ops.maximum(self.scan.sound_speed - sound_speed_offset, 1e-6)
        else:
            sound_speed = self.scan.sound_speed
        
        wavelength = sound_speed / self.scan.center_frequency
        
        # Positions: only scale if present
        positions = self.positions + (
            out["position_offset"] * wavelength
            if "position_offset" in out
            else ops.zeros_like(self.positions))

        # Element gains: only apply if present
        element_gains = (
            out["element_gains"] * 1e-3
            if "element_gains" in out
            else ops.zeros((self.scan.n_el,), dtype="float32")
        )
        
        # Initial times: only apply if present
        initial_times = self.scan.initial_times + (
            out["initial_times"] * 1e-6
            if "initial_times" in out
            else ops.zeros_like(self.scan.initial_times)
        )
        
        # Attenuation: only apply if present
        attenuation_coef = self.scan.attenuation_coef + (
            out["attenuation_coef"] * 1e-3
            if "attenuation_coef" in out
            else 0.0
        )
        
        factor = out["factor"] if "factor" in out else 1.0
        return positions, sound_speed, attenuation_coef, element_gains, initial_times, factor
    
    def forward(self, image, optvars, seed, freq_gaussian_probs=False, **kwargs):
        assert isinstance(optvars, dict), "optvars should be a dict"
        positions, sound_speed, attenuation_coef, element_gains, initial_times, factor = self.reparameterize_optvars(optvars)
        element_gains = ops.ones((self.scan.n_el,), dtype="float32") + element_gains

        assert len(image.shape) == 4, "Image should have shape (B, H, W, C)"
        image = ops.image.resize(image, self.shape, interpolation="bilinear")
        magnitudes = self.img_to_magnitude(image, positions)

        seed_el, seed_freq, seed_tx = split_seed(seed, 3)

        #sample el, freq and tx
        #randomly select n beams from self.beams
        tx_random = keras.random.uniform(shape=(self.beams.shape[0],), seed=seed_tx)
        tx_indices = ops.argsort(tx_random)[:self.n_tx_samples]
        tx_indices = ops.take(self.beams, tx_indices)

        def _gaussian_probs(_):
            freq_bins = ops.cast(ops.arange(self.n_freqs), "float32")
            center_freq_bin = ops.cast(self.n_freqs // 2, "float32")
            sigma = ops.cast(self.n_freqs / 8, "float32")
            gaussian_probs = ops.exp(-0.5 * ((freq_bins - center_freq_bin) / sigma) ** 2)
            return gaussian_probs / ops.sum(gaussian_probs)

        def _uniform_probs(_):
            return ops.ones((self.n_freqs,), dtype="float32") / self.n_freqs

        p = jax.lax.cond(ops.cast(freq_gaussian_probs, "bool"), _gaussian_probs, _uniform_probs, None)

        log_probs = ops.log(ops.maximum(p, 1e-10))
        gumbel = -ops.log(-ops.log(keras.random.uniform(shape=(self.n_freqs,), seed=seed_freq)))
        freq_random = log_probs + gumbel
        freq_indices = ops.argsort(freq_random)[:self.n_freq_samples]

        el_random = keras.random.uniform(shape=(self.scan.n_el,), seed=seed_el)
        el_indices = ops.argsort(el_random)[:self.n_el_samples]

        tx_indices = ops.sort(tx_indices)
        freq_indices = ops.sort(freq_indices)
        el_indices = ops.sort(el_indices)

        # Compute scatterer padding for chunking
        n_scat = self.positions.shape[0]
        chunk_size = self.scatterer_chunk_size
        n_scat_padded = n_scat + (chunk_size - (n_scat % chunk_size)) % chunk_size
        n_chunks = n_scat_padded // chunk_size  

        #pad and reshape positions and magnitudes for chunking
        positions = ops.pad(positions, ((0, n_scat_padded - n_scat), (0, 0)))
        position_chunks = ops.reshape(positions, (n_chunks, chunk_size, 3))
        magnitudes_padded = ops.pad(magnitudes, ((0, 0), (0, n_scat_padded - n_scat)))

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
                sound_speed=sound_speed,
                lens_sound_speed=self.scan.lens_sound_speed,
                lens_thickness=self.scan.lens_thickness,
                n_ax=self.scan.n_ax,
                center_frequency=self.scan.center_frequency,
                sampling_frequency=self.scan.sampling_frequency,
                t0_delays=self.scan.t0_delays,
                initial_times=initial_times,
                element_width=self.scan.element_width,
                attenuation_coef=attenuation_coef,
                tx_apodizations=self.scan.tx_apodizations,
            )
        
        @jax.checkpoint
        def accumulate_chunks(rf_accumulated, chunk):
            scat_chunk, amp_chunk = chunk
            rf_accumulated += simulate_chunk(scat_chunk, amp_chunk)
            return rf_accumulated, None

        @jax.checkpoint
        def process_frame(carry, frame_data):
            magnitude_chunks = ops.reshape(frame_data, (n_chunks, chunk_size))
            rf_init = ops.zeros((self.n_tx_samples, self.n_freq_samples, self.n_el_samples, 1), dtype='complex64')
            rf_accumulated, _ = ops.scan(accumulate_chunks, rf_init, (position_chunks, magnitude_chunks))
            return carry, rf_accumulated
        
        _, rf_data = ops.scan(process_frame, None, magnitudes_padded)
        element_gains = ops.take(element_gains, el_indices)
        return element_gains[None, None, :, None]*rf_data*factor, tx_indices, freq_indices, el_indices
        
    def __str__(self):
        return f"Simulator_Total with {self.n_tx_samples} tx samples, {self.n_freq_samples} freq samples, and {self.n_el_samples} el samples. Optimizing auxiliary variables"
    
    def transpose(self, data, *args, **kwargs):
        raise NotImplementedError("Transpose not implemented for Simulator_Total operator.")   

class Simulator_GD(Operator):
    """ Simulator that optimizes other auxiliary variables. 
    This may not depend on the dynamic range, because that is a visualization parameter, not part of the forward model """

    def __init__(self,
             scan,
             n_tx_samples,
             n_freq_samples,
             n_el_samples,
             scatterer_chunk_size = 256,
             n_equidistant_beams = 0,
             n_random_beams = 0,):
        super().__init__()
        self.scan = self.validate_scan(scan)
        self.shape = scan.grid.shape[:2]
        self.n_tx_samples = n_tx_samples
        self.n_freqs = 2**int(ops.ceil(ops.log2(self.scan.n_ax))) // 2 + 1
        self.n_freq_samples = n_freq_samples
        self.n_el_samples = n_el_samples
        self.scatterer_chunk_size = scatterer_chunk_size
        
        self.base_positions = self.scan.flatgrid
        self.base_sound_speed = jnp.asarray(self.scan.sound_speed, dtype=jnp.float32)
        self.base_initial_times = jnp.asarray(self.scan.initial_times, dtype=jnp.float32)
        self.base_attenuation_coef = jnp.asarray(self.scan.attenuation_coef, dtype=jnp.float32)
        
        self.beams = self.select_beams(n_equidistant_beams, n_random_beams)

    def validate_scan(self, scan):
        required_attrs = ["probe_geometry","apply_lens_correction","sound_speed","lens_sound_speed",
                          "lens_thickness","n_ax","center_frequency","sampling_frequency","t0_delays",
                          "initial_times","element_width","attenuation_coef","tx_apodizations"]
    
        if not hasattr(scan, "apply_lens_correction"):
            scan.apply_lens_correction = False  # Default to False if not provided
            log.warning("Scan object missing lens correction attributes. Defaulting to no lens correction.")
        if not hasattr(scan, "lens_sound_speed"):
            scan.lens_sound_speed = scan.sound_speed  # Default to same as sound speed
            log.warning("Scan object missing lens_sound_speed attribute. Defaulting to same as sound_speed.")
        if not hasattr(scan, "lens_thickness"):
            scan.lens_thickness = 0.0  # Default to no lens thickness
            log.warning("Scan object missing lens_thickness attribute. Defaulting to 0.0 (no lens).")
            
        for attr in required_attrs:
            if not hasattr(scan, attr):
                raise ValueError(f"Scan object is missing required attribute: {attr}")
        return scan
    
    def select_beams(self, n_equidistant, n_random):
        if n_equidistant == "all" or n_random == "all":
            return ops.arange(self.scan.n_tx, dtype="int32")
        
        if n_equidistant == 0 and n_random == 0:
            raise ValueError("At least one of n_equidistant_beams or n_random_beams must be greater than 0.")
        if n_equidistant > 0 and n_random > 0:
            raise ValueError("n_equidistant_beams and n_random_beams cannot both be greater than 0. Please choose one sampling strategy.")
        if n_equidistant > 0:
            #choose n equidistant transmits from the scan
            beams = ops.linspace(0, self.scan.n_tx-1, n_equidistant, dtype="int32")
        if n_random > 0:
            #choose n random transmits from the scan
            beams = ops.random.shuffle(ops.arange(self.scan.n_tx, dtype="int32"))[:n_random]
        if self.n_tx_samples > len(beams):
            raise ValueError(f"n_tx_samples ({self.n_tx_samples}) cannot be greater than the number of selected beams ({len(beams)}). Please adjust n_tx_samples or the beam selection strategy.")
        return beams
    
    def apply_optvars(self, optvars):
        """Apply optimization variables to baseline scan/scatterer parameters.

        Supports both offset-style and absolute overrides for compatibility.
        """
        if optvars is None:
            raise ValueError("optvars cannot be None. Please provide an empty dict if no optimization variables are used.")
        else:
            out = dict(optvars)
        
        # Sound speed in m/s.
        if "sound_speed_abs" in out:
            sound_speed = ops.cast(out["sound_speed_abs"], "float32")
        elif "sound_speed_offset" in out:
            sound_speed = ops.cast(self.base_sound_speed - out["sound_speed_offset"], "float32")
        else:
            sound_speed = self.base_sound_speed
        sound_speed = ops.maximum(sound_speed, 1e-6)
        
        wavelength = sound_speed / self.scan.center_frequency
        
        # Position offsets are interpreted in wavelengths if position_offset_wl is used.
        if "positions_abs" in out:
            positions = out["positions_abs"]
        elif "position_offset_wl" in out:
            positions = self.base_positions + out["position_offset_wl"] * wavelength
        elif "position_offset" in out:
            positions = self.base_positions + out["position_offset"] * wavelength
        else:
            positions = self.base_positions

        # Per-element gain.
        if "element_gains_abs" in out:
            element_gains = ops.cast(out["element_gains_abs"], "float32")
        elif "element_gain_abs" in out:
            element_gains = ops.cast(out["element_gain_abs"], "float32")
        elif "element_gains" in out:
            # Backward compatibility path: treat element_gains as small offsets.
            element_gains = ops.ones((self.scan.n_el,), dtype="float32") + out["element_gains"] * 1e-3
        else:
            element_gains = ops.ones((self.scan.n_el,), dtype="float32")
        
        # Initial time shift in seconds.
        if "initial_times_abs" in out:
            initial_times = out["initial_times_abs"]
        elif "initial_time_shift" in out:
            initial_times = self.base_initial_times + out["initial_time_shift"]
        else:
            initial_times = self.base_initial_times

        # Attenuation: only apply if present
        attenuation_coef = self.base_attenuation_coef + (out["attenuation_coef"] if "attenuation_coef" in out else 0.0)
        
        factor = out["factor"] if "factor" in out else 1.0

        return positions, sound_speed, attenuation_coef, element_gains, initial_times, factor
    
    def forward(self, image, optvars, seed, freq_gaussian_probs=False, **kwargs):
        if optvars is None:
            optvars = {}
        assert isinstance(optvars, dict), "optvars should be a dict"
        positions, sound_speed, attenuation_coef, element_gains, initial_times, factor = self.apply_optvars(optvars)

        assert len(image.shape) == 4, "Image should have shape (B, H, W, C)"
        n_frames, *img_shape = image.shape
        magnitudes = ops.reshape(image, (n_frames, -1))

        seed_el, seed_freq, seed_tx = split_seed(seed, 3)

        #sample el, freq and tx
        #randomly select n beams from self.beams
        tx_random = keras.random.uniform(shape=(self.beams.shape[0],), seed=seed_tx)
        tx_indices = ops.argsort(tx_random)[:self.n_tx_samples]
        tx_indices = ops.take(self.beams, tx_indices)

        def _gaussian_probs(_):
            freq_bins = ops.cast(ops.arange(self.n_freqs), "float32")
            center_freq_bin = ops.cast(self.n_freqs // 2, "float32")
            sigma = ops.cast(self.n_freqs / 8, "float32")
            gaussian_probs = ops.exp(-0.5 * ((freq_bins - center_freq_bin) / sigma) ** 2)
            return gaussian_probs / ops.sum(gaussian_probs)

        def _uniform_probs(_):
            return ops.ones((self.n_freqs,), dtype="float32") / self.n_freqs

        p = jax.lax.cond(ops.cast(freq_gaussian_probs, "bool"), _gaussian_probs, _uniform_probs, None)

        log_probs = ops.log(ops.maximum(p, 1e-10))
        gumbel = -ops.log(-ops.log(keras.random.uniform(shape=(self.n_freqs,), seed=seed_freq)))
        freq_random = log_probs + gumbel
        freq_indices = ops.argsort(freq_random)[:self.n_freq_samples]

        el_random = keras.random.uniform(shape=(self.scan.n_el,), seed=seed_el)
        el_indices = ops.argsort(el_random)[:self.n_el_samples]

        tx_indices = ops.sort(tx_indices)
        freq_indices = ops.sort(freq_indices)
        el_indices = ops.sort(el_indices)

        # Compute scatterer padding for chunking
        n_scat = positions.shape[0]
        chunk_size = self.scatterer_chunk_size
        n_scat_padded = n_scat + (chunk_size - (n_scat % chunk_size)) % chunk_size
        n_chunks = n_scat_padded // chunk_size  

        #pad and reshape positions and magnitudes for chunking
        positions = ops.pad(positions, ((0, n_scat_padded - n_scat), (0, 0)))
        position_chunks = ops.reshape(positions, (n_chunks, chunk_size, 3))
        magnitudes_padded = ops.pad(magnitudes, ((0, 0), (0, n_scat_padded - n_scat)))

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
                sound_speed=sound_speed,
                lens_sound_speed=self.scan.lens_sound_speed,
                lens_thickness=self.scan.lens_thickness,
                n_ax=self.scan.n_ax,
                center_frequency=self.scan.center_frequency,
                sampling_frequency=self.scan.sampling_frequency,
                t0_delays=self.scan.t0_delays,
                initial_times=initial_times,
                element_width=self.scan.element_width,
                attenuation_coef=attenuation_coef,
                tx_apodizations=self.scan.tx_apodizations,
            )
        
        @jax.checkpoint
        def accumulate_chunks(rf_accumulated, chunk):
            scat_chunk, amp_chunk = chunk
            rf_accumulated += simulate_chunk(scat_chunk, amp_chunk)
            return rf_accumulated, None

        @jax.checkpoint
        def process_frame(carry, frame_data):
            magnitude_chunks = ops.reshape(frame_data, (n_chunks, chunk_size))
            rf_init = ops.zeros((self.n_tx_samples, self.n_freq_samples, self.n_el_samples, 1), dtype='complex64')
            rf_accumulated, _ = ops.scan(accumulate_chunks, rf_init, (position_chunks, magnitude_chunks))
            return carry, rf_accumulated
        
        _, rf_data = ops.scan(process_frame, None, magnitudes_padded)
        element_gains = ops.take(element_gains, el_indices)
        return element_gains[None, None, None, :, None]*rf_data*factor, tx_indices, freq_indices, el_indices
        
    def __str__(self):
        return f"Simulator_GD with {self.n_tx_samples} tx samples, {self.n_freq_samples} freq samples, and {self.n_el_samples} el samples. Optimizing auxiliary variables"
    
    def transpose(self, data, *args, **kwargs):
        raise NotImplementedError("Transpose not implemented for Simulator_GD operator.")   
