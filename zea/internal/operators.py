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
from zea.simulator_fourier import simulate_rf
import jax
import jax.numpy as jnp
from zea.display import frustum_convert_xz2rt, map_coordinates

from zea.simulator_time import simulate_partial_rf_data

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


@operator_registry(name="linear_interp")
class LinearInterpOperator(Operator):
    r"""Linear interpolation operator for ultrasound image degradation modeling.

    The linear interpolation operator models the forward process of blending
    two components (e.g., clean tissue and haze) to simulate the observed measurement that
    contains contributions from both.

    .. math::

        \mathbf{y} = (1 - \alpha) \mathbf{x} + \alpha \mathbf{h}

    where:

    - :math:`\mathbf{x}` is the first component
    - :math:`\mathbf{h}` is the second component
    - :math:`\alpha \in [0, 1]` is the blending factor controlling the mixing ratio
    - :math:`\mathbf{y}` is the observed (blended) measurement

    Note:
        Compared to other operators, a second component must be provided as an
        additional argument to both :meth:`forward` and :meth:`transpose` methods.

    See Also:
        - :class:`~zea.models.diffusion.NuclearDiffusion`: Uses this operator for posterior sampling
        - :doc:`../notebooks/models/nuclear_dehazing_example`: Example notebook

    Example:
        .. doctest::

            from zea.internal.operators import LinearInterpOperator
            import numpy as np

            operator = LinearInterpOperator()
            tissue = np.random.randn(64, 64, 1)
            haze = np.random.randn(64, 64, 1)

            # Create hazy measurement
            measurement = operator.forward(tissue, haze, haze_level=0.5)
    """

    def forward(self, data1, data2, blend_level: float = 0.5):
        r"""Apply linear interpolation to blend two components.

        Args:
            data1: First component.
            data2: Second component.
            blend_level: Blending factor :math:`\alpha \in [0, 1]`. Higher values
                mean more contribution from the second component. Default is 0.5.

        Returns:
            Blended measurement.
        """
        out = (1 - blend_level) * data1 + blend_level * data2
        return out

    def transpose(self, data, blend_level: float = 0.5):
        """Transpose operator."""
        return (1 - blend_level) * data

    def __str__(self):
        return "y = (1-α)x + αh"


def _validate_parameters(scan):
    """Validate a scan object and fill in lens-correction defaults.

    Shared between `Simulator` and `Simulator_Total` so that both operators see
    identical `lens_sound_speed` / `lens_thickness` values for the same scan.
    """
    required_attrs = ["probe_geometry","apply_lens_correction","sound_speed","lens_sound_speed",
                      "lens_thickness","n_ax","center_frequency","sampling_frequency","t0_delays",
                      "initial_times","element_width","attenuation_coef","tx_apodizations"]

    if not hasattr(scan, "apply_lens_correction"):
        scan.apply_lens_correction = False  # Default to False if not provided
        scan.lens_sound_speed = scan.sound_speed  # Default to same as sound speed
        scan.lens_thickness = 0.0  # Default to no lens thickness
        log.warning("Scan object missing lens correction attributes. Defaulting to no lens correction.")

    if scan.apply_lens_correction is False:
        #add these attributes, because they are probably not declared
        scan.lens_sound_speed = scan.sound_speed
        scan.lens_thickness = 0.0

    if not hasattr(scan, "element_width"):
        #default to tiny gap between elements, so that the forward model is still valid
        scan.element_width = scan.aperture_size.max() / scan.n_el * 0.95
        log.warning(f"Scan object missing 'element_width' attribute. Got {scan.element_width}.")

    for attr in required_attrs:
        if not hasattr(scan, attr):
            #get all missing attributes and raise them in one error message
            missing_attrs = [attr for attr in required_attrs if not hasattr(scan, attr)]
            raise ValueError(f"Scan object is missing required attributes: {missing_attrs}")
    return scan


def _sample_indices(n_el, n_freqs, beams, n_tx_samples, n_freq_samples, n_el_samples,
                     seed_el, seed_freq, seed_tx, freq_gaussian_probs):
    """Sample element, frequency and transmit indices.

    Shared between `Simulator` and `Simulator_Total` so both operators sample
    identical indices given the same seeds.
    """

    def _gaussian_probs(_):
        freq_bins = ops.cast(ops.arange(n_freqs), "float32")
        center_freq_bin = ops.cast(n_freqs // 2, "float32")
        sigma = ops.cast(n_freqs / 8, "float32")
        gaussian_probs = ops.exp(-0.5 * ((freq_bins - center_freq_bin) / sigma) ** 2)
        return gaussian_probs / ops.sum(gaussian_probs)

    def _uniform_probs(_):
        return ops.ones((n_freqs,), dtype="float32") / n_freqs

    if getattr(freq_gaussian_probs, "ndim", 0) == 1:
        # Explicit per-bin sampling probabilities (e.g. proportional to the measured
        # spectrum's RMS): importance-sample the loss where the data has energy,
        # instead of the fixed fs/4-centred Gaussian below.
        assert freq_gaussian_probs.shape[0] == n_freqs, (
            f"freq probs shape {freq_gaussian_probs.shape} != n_freqs {n_freqs}"
        )
        p = ops.cast(freq_gaussian_probs, "float32")
        p = p / ops.sum(p)
    else:
        p = jax.lax.cond(ops.cast(freq_gaussian_probs, "bool"), _gaussian_probs, _uniform_probs, None)

    log_probs = ops.log(ops.maximum(p, 1e-10))
    gumbel = -ops.log(-ops.log(keras.random.uniform(shape=(n_freqs,), seed=seed_freq)))
    freq_random = log_probs + gumbel
    freq_indices = ops.argsort(freq_random)[-n_freq_samples:]

    tx_random = keras.random.uniform(shape=(len(beams),), seed=seed_tx)
    tx_indices = ops.argsort(tx_random)[:n_tx_samples]

    el_random = keras.random.uniform(shape=(n_el,), seed=seed_el)
    el_indices = ops.argsort(el_random)[:n_el_samples]

    el_indices = ops.sort(el_indices)
    freq_indices = ops.sort(freq_indices)
    tx_indices = ops.sort(tx_indices)

    tx_indices = ops.take(beams, tx_indices)
    return el_indices, freq_indices, tx_indices


class Simulator(Operator):
    """
    Simulator operator. This consists of 2 steps:
    - Converting image into scatter poitns
    - Simulating RF data from scatter points. 
    To speed up DPS, only a few indices are chosen per axis. They are also returned.

    The scan object contains all parameters of the forward model. 
    In case we want to choose from the same transmits for each image, we can set n_equidistant_beams or n_random_beams to a non-zero value.
    """
    def __init__(self,
                 parameters,
                 n_tx_samples,
                 n_freq_samples,
                 n_el_samples,
                 scatterer_chunk_size = 256,
                 n_equidistant_beams = None,
                 n_random_beams = None,
                 grid = None,
                 n_ax_min = 0,
                 magnitude_range = (-320, 0),
                 waveform = None,
                 waveform_sampling_frequency = None):
        super().__init__()
        self.parameters = self.validate_parameters(parameters)
        self.n_tx_samples = parameters.n_tx if n_tx_samples is None else n_tx_samples
        self.n_freqs = 2**int(ops.ceil(ops.log2(self.parameters.n_ax-n_ax_min))) // 2 + 1
        self.n_freq_samples = self.n_freqs if n_freq_samples is None else n_freq_samples
        self.n_el_samples = parameters.n_el if n_el_samples is None else n_el_samples
        self.scatterer_chunk_size = scatterer_chunk_size
        # Optional measured transmit waveform. When None, simulate_rf uses the
        # default analytic Hann pulse (previous behaviour).
        self.waveform = waveform
        self.waveform_sampling_frequency = waveform_sampling_frequency
        # dB range that image values in [-1, 1] map to before exp() -> linear amplitude.
        # A very wide range (e.g. 320 dB) makes the forward model blind to all but the
        # brightest pixels (exp(-16) ~ 0), so its gradient w.r.t. dark pixels vanishes and
        # DPS cannot reconstruct them. Match this to the data's actual dynamic range.
        self.magnitude_range = tuple(magnitude_range)
        self.beams = self.select_beams(n_equidistant_beams, n_random_beams)
        # Scatterer grid resolution (grid_size_z, grid_size_x). The input image is
        # resized to this in `forward`, so the image resolution (e.g. the diffusion
        # model's native 112x112) is decoupled from the simulation grid resolution.
        # Mirrors Simulator_Total; a resize to the image's own shape is a no-op.
        self.shape = parameters.grid.shape[:2]
        self.positions = parameters.flatgrid if grid is None else grid if grid.ndim==2 else ValueError(f"grid should be 2D, but got shape {grid.shape}")
        self.initial_time_offset = n_ax_min / self.parameters.sampling_frequency
        self.n_ax_min = n_ax_min

    def select_beams(self, n_equidistant, n_random):
        if n_equidistant is None and n_random is None:
            beams = ops.arange(self.parameters.n_tx, dtype="int32")
        else:
            log.warning("Beam selection is deprecated. Please use parameters.set_transmits() to select transmits instead.")
            log.info("selecting all beams")
            beams = ops.arange(self.parameters.n_tx, dtype="int32")
        return beams
    
    def validate_parameters(self, parameters):
        return _validate_parameters(parameters)

    def image_to_magnitudes(self, image, n_frames):
        # check for nans
        if jnp.any(jnp.isnan(image)):
            log.warning("Image contains NaNs. Replacing with 0.")
            image = ops.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
        assert image.max() < 1.01 and image.min() > -1.01, f"Image values should be in the range [-1, 1]. got (min, max) = ({image.min()}, {image.max()})"
        
        # Translate to dB range (e.g., [-60, 0])
        image = translate(image, range_from=(-1,1), range_to=self.magnitude_range)
        image = ops.reshape(image, (n_frames, -1))
        min_db = self.magnitude_range[0]
        # Convert to linear amplitude and shift the noise floor to exactly 0.0
        image_lin = ops.power(10.0, image / 20.0) - ops.power(10.0, min_db / 20.0)
        
        # Clip tiny negative values that might occur due to floating-point precision
        image_lin = ops.relu(image_lin)
            
        return image_lin

    def _get_indices(self, seed_el, seed_freq, seed_tx, freq_gaussian_probs):
        return _sample_indices(
            self.parameters.n_el, self.n_freqs, self.beams,
            self.n_tx_samples, self.n_freq_samples, self.n_el_samples,
            seed_el, seed_freq, seed_tx, freq_gaussian_probs,
        )

    def forward(self, image, seed, freq_gaussian_probs=False, linearized=False):
        seed_el, seed_freq, seed_tx = split_seed(seed, 3)

        if linearized:
            # `image` is already a (n_frames, n_scat) array of (area-weighted) scatterer
            # magnitudes — skip the resize + image_to_magnitudes mapping entirely. May be
            # complex (per-scatterer phase); the rest of the pipeline is complex-safe.
            assert image.ndim == 2, (
                "linearized=True expects magnitudes of shape (n_frames, n_scat), "
                f"got {image.shape}"
            )
            n_frames = image.shape[0]
            magnitudes = image
        else:
            assert len(image.shape) == 4, "Image should have shape (B, H, W, C)"
            # Resample the image onto the scatterer grid (no-op when already equal).
            image = ops.image.resize(image, self.shape, interpolation="bilinear")
            n_frames = image.shape[0]
            magnitudes = self.image_to_magnitudes(image, n_frames)

        el_indices, freq_indices, tx_indices = self._get_indices(seed_el, seed_freq, seed_tx, freq_gaussian_probs)

        # Compute scatterer padding for chunking
        n_scat = self.positions.shape[0]
        chunk_size = self.scatterer_chunk_size
        n_scat_padded = n_scat + (chunk_size - (n_scat % chunk_size)) % chunk_size
        n_chunks = n_scat_padded // chunk_size

        # Pad and pre-reshape positions into (n_chunks, chunk_size, 3)
        positions_padded = ops.pad(self.positions, ((0, n_scat_padded - n_scat), (0, 0)))
        position_chunks = ops.reshape(positions_padded, (n_chunks, chunk_size, 3))
        magnitudes_padded = ops.pad(magnitudes, ((0, 0), (0, n_scat_padded - n_scat)))
        # (n_chunks, n_frames, chunk_size): each scan step processes all frames for one
        # scatterer chunk, so the (frame-independent) kernel is built once and contracted
        # against every frame's magnitudes in a single pass.
        magnitude_chunks = ops.transpose(
            ops.reshape(magnitudes_padded, (n_frames, n_chunks, chunk_size)), (1, 0, 2)
        )

        def simulate_chunk(pos_chunk, mag_chunk):
            """Simulate RF for a single scatterer chunk (all frames). mag_chunk: (n_frames, chunk)."""
            return simulate_rf(
                scatterer_positions=pos_chunk,
                scatterer_magnitudes=mag_chunk,
                el_indices=el_indices,
                freq_indices=freq_indices,
                tx_indices=tx_indices,
                probe_geometry=self.parameters.probe_geometry,
                apply_lens_correction=self.parameters.apply_lens_correction,
                sound_speed=self.parameters.sound_speed,
                lens_sound_speed=self.parameters.lens_sound_speed,
                lens_thickness=self.parameters.lens_thickness,
                n_ax=self.parameters.n_ax - self.n_ax_min,
                center_frequency=self.parameters.center_frequency,
                sampling_frequency=self.parameters.sampling_frequency,
                t0_delays=self.parameters.t0_delays,
                initial_times=self.parameters.initial_times+self.initial_time_offset,
                element_width=self.parameters.element_width,
                attenuation_coef=self.parameters.attenuation_coef,
                tx_apodizations=self.parameters.tx_apodizations,
                tgc_gain_curve = self.parameters.tgc_gain_curve,
                waveform=self.waveform,
                waveform_sampling_frequency=self.waveform_sampling_frequency,
            )

        # No @jax.checkpoint: magnitudes are factored out of the kernel (see simulate_rf), so
        # the only intermediate the backward pass needs is the small post-`tx_el`-sum kernel,
        # which `scan` stores per chunk (~GBs, well within VRAM). This avoids recomputing the
        # full forward simulation in the gradient pass.
        def accumulate_chunks(rf_accumulated, chunk):
            pos_chunk, mag_chunk = chunk
            rf_accumulated = rf_accumulated + simulate_chunk(pos_chunk, mag_chunk)
            return rf_accumulated, None

        rf_init = ops.zeros(
            (n_frames, self.n_tx_samples, self.n_freq_samples, self.n_el_samples, 1),
            dtype='complex64',
        )
        rf_data, _ = ops.scan(accumulate_chunks, rf_init, (position_chunks, magnitude_chunks))

        return rf_data, tx_indices, freq_indices, el_indices

    def __str__(self):
        return f"Simulator with {self.n_tx_samples} tx samples, {self.n_freq_samples} freq samples, and {self.n_el_samples} el samples. Not optimizing auxiliary variables."

    def transpose(self, data, *args, **kwargs):
        raise NotImplementedError("Transpose not implemented for Simulator operator.")
    
# Other operators were removed. They were not part of the original python package, but self made. AI tools confuse this as being "good code" but it is not good code. 