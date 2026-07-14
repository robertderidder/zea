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
        #check for nans
        if jnp.any(jnp.isnan(image)):
            log.warning("Image contains NaNs. Replacing with 0.")
            image = ops.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
        assert image.max() < 1.01 and image.min() > -1.01, f"Image values should be in the range [-1, 1]. got (min, max) = ({image.min()}, {image.max()})"
        image = translate(image, range_from=(-1,1), range_to=self.magnitude_range)
        image = ops.reshape(image, (n_frames, -1))
        image_lin  = ops.exp(image/20) #Means values between 0 and 6.4.
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
    

class Simulator_Total(Operator):
    """Simulator operator that additionally optimizes scatterer positions and sound speed.

    This is a more flexible version of `Simulator`: in addition to the scatterer
    amplitudes (derived from the input image, same as `Simulator`), it supports
    optimizing ``optvars["position_offset"]`` (a per-scatterer position offset, in
    wavelengths) and ``optvars["sound_speed_offset"]`` (a global sound speed offset).

    When ``optvars["position_offset"]`` is all zeros and ``optvars["sound_speed_offset"]``
    is ``0.0`` (the defaults), this operator produces the same result as `Simulator`.
    """

    def __init__(self,
             scan,
             n_tx_samples,
             n_freq_samples,
             n_el_samples,
             scatterer_chunk_size = 256,
             position_offset_wl = 0.0,
             sound_speed_offset_scale = 100.0,
             n_equidistant_beams = 0,
             n_random_beams = 0,
             grid = None,
             n_ax_min = 0,
             magnitude_range = (-320, 0),
             enable_wavelength_scaling = False,
             waveform = None,
             waveform_sampling_frequency = None):
        super().__init__()
        self.parameters = self.validate_parameters(scan)
        self.shape = scan.grid.shape[:2]
        # When True, the scatterer grid is parameterized in wavelengths and scales with
        # the (optimized) sound speed: a higher sound speed moves every scatterer outward
        # so round-trip travel times stay ~constant. This keeps the loss landscape in
        # `sound_speed_offset` smooth instead of phase-wrapping, which is what makes joint
        # sound-speed optimization converge (cf. off-grid-ultrasound reparameterize_scat_pos).
        self.enable_wavelength_scaling = enable_wavelength_scaling
        # Optional measured transmit waveform (see Simulator). None -> analytic pulse.
        self.waveform = waveform
        self.waveform_sampling_frequency = waveform_sampling_frequency
        self.n_tx_samples = scan.n_tx if n_tx_samples is None else n_tx_samples
        self.n_freqs = 2**int(ops.ceil(ops.log2(self.parameters.n_ax-n_ax_min))) // 2 + 1
        self.n_freq_samples = self.n_freqs if n_freq_samples is None else n_freq_samples
        self.n_el_samples = scan.n_el if n_el_samples is None else n_el_samples
        self.scatterer_chunk_size = scatterer_chunk_size
        self.sound_speed_offset_scale = sound_speed_offset_scale
        # See Simulator.magnitude_range: dB range mapped to before exp() -> amplitude.
        self.magnitude_range = tuple(magnitude_range)
        base_positions = scan.flatgrid if grid is None else grid
        self.positions = base_positions + 2 * position_offset_wl * (keras.random.uniform(shape=base_positions.shape, seed=42)-ops.ones_like(base_positions)*0.5)* scan.wavelength
        # Reference wavelength at the unperturbed sound speed, and the grid expressed in
        # wavelength units. When sound speed is optimized, the physical positions are
        # rebuilt as positions_wl * (new) wavelength (see reparameterize_optvars). At the
        # default sound_speed_offset == 0 this returns exactly self.positions, so the
        # operator still matches Simulator with default optvars.
        self.base_wavelength = self.parameters.sound_speed / self.parameters.center_frequency
        self.positions_wl = self.positions / self.base_wavelength
        self.cell_size = ops.convert_to_tensor(scan.flatgrid_cell_size, dtype="float32")
        # Normalize so the mean weight is 1: preserves the *relative* per-scatterer
        # area weighting (e.g. far-field cells vs. near-apex cells on a polar grid)
        # while keeping the overall output scale compatible with omega/eps, which
        # were tuned without any area weighting.
        self.beams = self.select_beams(n_equidistant_beams, n_random_beams)
        self.initial_time_offset = n_ax_min / self.parameters.sampling_frequency
        self.n_ax_min = n_ax_min

    def validate_parameters(self, scan):
        return _validate_parameters(scan)

    def select_beams(self, n_equidistant, n_random):
        assert isinstance(n_equidistant, int) or n_equidistant == "all", "n_equidistant_beams should be an integer or 'all'"
        assert isinstance(n_random, int) or n_random == "all", "n_random_beams should be an integer or 'all'"
        if n_equidistant == "all" or n_random == "all":
            return ops.arange(self.parameters.n_tx, dtype="int32")
        elif n_equidistant == 0 and n_random == 0:
            raise ValueError("At least one of n_equidistant_beams or n_random_beams must be greater than 0.")
        elif n_equidistant > 0 and n_random > 0:
            raise ValueError("n_equidistant_beams and n_random_beams cannot both be greater than 0. Please choose one sampling strategy.")
        elif n_equidistant > 0:
            #choose n equidistant transmits from the scan
            beams = ops.linspace(0, self.parameters.n_tx-1, n_equidistant, dtype="int32")
        elif n_random > 0:
            #choose n random transmits from the scan
            beams = ops.random.shuffle(ops.arange(self.parameters.n_tx, dtype="int32"))[:n_random]
        else:
            raise ValueError("Invalid beam selection strategy. Please check n_equidistant_beams and n_random_beams parameters.")

        if self.n_tx_samples > len(beams):
            raise ValueError(f"n_tx_samples ({self.n_tx_samples}) cannot be greater than the number of selected beams ({len(beams)}). Please adjust n_tx_samples or the beam selection strategy.")
        return beams

    def image_to_magnitudes(self, image, n_frames):
        """Convert a (resized) image into scatterer magnitudes. Mirrors `Simulator.image_to_magnitudes`."""
        image = translate(image, range_from=(-1,1), range_to=self.magnitude_range)
        image = ops.reshape(image, (n_frames, -1))
        image_lin = ops.exp(image/20)
        # Riemann-sum area weighting, see `Simulator.image_to_magnitudes`.
        return image_lin

    def reparameterize_optvars(self, optvars):
        """Reparameterize the position and sound speed optimization variables.

        With ``optvars["position_offset"]`` all zeros and ``optvars["sound_speed_offset"]``
        equal to ``0.0`` (the defaults), this reduces to `self.positions` and
        `self.parameters.sound_speed`, matching `Simulator`.

        When ``enable_wavelength_scaling`` is True, the whole scatterer grid is defined in
        wavelengths and rebuilt at the current (optimized) wavelength, so increasing the
        sound speed moves every scatterer outward in proportion. This keeps the dominant
        round-trip travel time ``dist/c`` ~constant as ``c`` varies, so the measurement
        error is smooth in ``sound_speed_offset`` (no phase wrapping) and joint sound-speed
        optimization converges. With it False the grid is fixed in metres and only the
        per-scatterer ``position_offset`` moves (the previous, poorly-conditioned behaviour).
        """
        out = dict(optvars)

        if "sound_speed_offset" in out:
            sound_speed_offset = out["sound_speed_offset"] * self.sound_speed_offset_scale
            sound_speed = ops.maximum(self.parameters.sound_speed - sound_speed_offset, 1e-6)
        else:
            sound_speed = self.parameters.sound_speed

        wavelength = sound_speed / self.parameters.center_frequency

        # Per-scatterer offset, in wavelengths (zeros by default).
        position_offset_wl = (
            out["position_offset"] if "position_offset" in out
            else ops.zeros_like(self.positions))

        if self.enable_wavelength_scaling:
            # Grid (and offset) are in wavelengths -> scale to metres at the current
            # wavelength. At sound_speed_offset == 0 this is exactly self.positions
            # (+ offset * base_wavelength), preserving the Simulator equivalence.
            positions = (self.positions_wl + position_offset_wl) * wavelength
        else:
            positions = self.positions + position_offset_wl * wavelength

        return positions, sound_speed

    def forward(self, image, optvars, seed, freq_gaussian_probs=False, **kwargs):
        assert isinstance(optvars, dict), "optvars should be a dict"
        positions, sound_speed = self.reparameterize_optvars(optvars)

        assert len(image.shape) == 4, "Image should have shape (B, H, W, C)"
        image = ops.image.resize(image, self.shape, interpolation="bilinear")
        n_frames = image.shape[0]
        magnitudes = self.image_to_magnitudes(image, n_frames)

        seed_el, seed_freq, seed_tx = split_seed(seed, 3)
        el_indices, freq_indices, tx_indices = _sample_indices(
            self.parameters.n_el, self.n_freqs, self.beams,
            self.n_tx_samples, self.n_freq_samples, self.n_el_samples,
            seed_el, seed_freq, seed_tx, freq_gaussian_probs,
        )

        # Compute scatterer padding for chunking
        n_scat = positions.shape[0]
        chunk_size = self.scatterer_chunk_size
        n_scat_padded = n_scat + (chunk_size - (n_scat % chunk_size)) % chunk_size
        n_chunks = n_scat_padded // chunk_size

        #pad and reshape positions, cell sizes and magnitudes for chunking
        positions_padded = ops.pad(positions, ((0, n_scat_padded - n_scat), (0, 0)))
        position_chunks = ops.reshape(positions_padded, (n_chunks, chunk_size, 3))
        magnitudes_padded = ops.pad(magnitudes, ((0, 0), (0, n_scat_padded - n_scat)))
        # (n_chunks, n_frames, chunk_size): all frames per chunk (see Simulator.forward).
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
                sound_speed=sound_speed,
                lens_sound_speed=self.parameters.lens_sound_speed,
                lens_thickness=self.parameters.lens_thickness,
                n_ax=self.parameters.n_ax - self.n_ax_min,
                center_frequency=self.parameters.center_frequency,
                sampling_frequency=self.parameters.sampling_frequency,
                t0_delays=self.parameters.t0_delays,
                initial_times=self.parameters.initial_times + self.initial_time_offset,
                element_width=self.parameters.element_width,
                attenuation_coef=self.parameters.attenuation_coef,
                tx_apodizations=self.parameters.tx_apodizations,
                waveform=self.waveform,
                waveform_sampling_frequency=self.waveform_sampling_frequency,
                tgc_gain_curve=self.parameters.tgc_gain_curve,
            )

        # Kept @jax.checkpoint here (unlike Simulator): this operator also optimizes
        # `position_offset` / `sound_speed_offset`, whose gradients flow THROUGH the kernel
        # build (positions/sound_speed enter the transcendental terms), so the backward needs
        # the large per-chunk intermediates. Rematerializing them per chunk bounds memory on
        # large grids. Magnitude factoring + frame batching still apply.
        @jax.checkpoint
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
        return f"Simulator_Total with {self.n_tx_samples} tx samples, {self.n_freq_samples} freq samples, and {self.n_el_samples} el samples. Optimizing position_offset and sound_speed_offset."

    def transpose(self, data, *args, **kwargs):
        raise NotImplementedError("Transpose not implemented for Simulator_Total operator.")

class Simlator_Time(Operator):
    """Simulator operator that additionally optimizes scatterer positions and sound speed.

    This is a more flexible version of `Simulator`: in addition to the scatterer
    amplitudes (derived from the input image, same as `Simulator`), it supports
    optimizing ``optvars["position_offset"]`` (a per-scatterer position offset, in
    wavelengths) and ``optvars["sound_speed_offset"]`` (a global sound speed offset).

    When ``optvars["position_offset"]`` is all zeros and ``optvars["sound_speed_offset"]``
    is ``0.0`` (the defaults)+, this operator produces the same result as `Simulator`.
    """

    def __init__(self,
             scan,
             n_tx_samples,
             n_ax_samples,
             n_el_samples,
             scatterer_chunk_size = 256,
             position_offset_wl = 0.0,
             sound_speed_offset_scale = 100.0,
             n_equidistant_beams = 0,
             n_random_beams = 0,
             grid = None,
             n_ax_min = 0):
        super().__init__()
        self.parameters = self.validate_parameters(scan)
        self.shape = scan.grid.shape[:2]
        self.n_tx_samples = scan.n_tx if n_tx_samples is None else n_tx_samples
        self.n_ax_samples = scan.n_ax if n_ax_samples is None else n_ax_samples
        self.n_el_samples = scan.n_el if n_el_samples is None else n_el_samples
        self.scatterer_chunk_size = scatterer_chunk_size
        self.sound_speed_offset_scale = sound_speed_offset_scale
        base_positions = scan.flatgrid if grid is None else grid
        self.positions = base_positions + 2 * position_offset_wl * (keras.random.uniform(shape=base_positions.shape, seed=42)-ops.ones_like(base_positions)*0.5)* scan.wavelength
        # Normalize so the mean weight is 1: preserves the *relative* per-scatterer
        # area weighting (e.g. far-field cells vs. near-apex cells on a polar grid)
        # while keeping the overall output scale compatible with omega/eps, which
        # were tuned without any area weighting.
        self.beams = self.select_beams(n_equidistant_beams, n_random_beams)
        self.initial_time_offset = n_ax_min / self.parameters.sampling_frequency
        self.n_ax_min = n_ax_min

    def validate_parameters(self, scan):
        return _validate_parameters(scan)

    def select_beams(self, n_equidistant, n_random):
        if n_equidistant == "all" or n_random == "all":
            return ops.arange(self.parameters.n_tx, dtype="int32")

        if n_equidistant == 0 and n_random == 0:
            raise ValueError("At least one of n_equidistant_beams or n_random_beams must be greater than 0.")
        if n_equidistant > 0 and n_random > 0:
            raise ValueError("n_equidistant_beams and n_random_beams cannot both be greater than 0. Please choose one sampling strategy.")
        if n_equidistant > 0:
            #choose n equidistant transmits from the scan
            beams = ops.linspace(0, self.parameters.n_tx-1, n_equidistant, dtype="int32")
        if n_random > 0:
            #choose n random transmits from the scan
            beams = ops.random.shuffle(ops.arange(self.parameters.n_tx, dtype="int32"))[:n_random]
        if self.n_tx_samples > len(beams):
            raise ValueError(f"n_tx_samples ({self.n_tx_samples}) cannot be greater than the number of selected beams ({len(beams)}). Please adjust n_tx_samples or the beam selection strategy.")
        return beams

    def image_to_magnitudes(self, image, n_frames):
        """Convert a (resized) image into scatterer magnitudes. Mirrors `Simulator.image_to_magnitudes`."""
        image = translate(image, range_from=(-1,1), range_to=(-320,0))
        image = ops.reshape(image, (n_frames, -1))
        image_lin = ops.exp(image/20)
        # Riemann-sum area weighting, see `Simulator.image_to_magnitudes`.
        return image_lin * self.cell_area[None, :]

    def reparameterize_optvars(self, optvars):
        """Reparameterize the position and sound speed optimization variables.

        With ``optvars["position_offset"]`` all zeros and ``optvars["sound_speed_offset"]``
        equal to ``0.0`` (the defaults), this reduces to `self.positions` and
        `self.parameters.sound_speed`, matching `Simulator`.
        """
        out = dict(optvars)

        if "sound_speed_offset" in out:
            sound_speed_offset = out["sound_speed_offset"] * self.sound_speed_offset_scale
            sound_speed = ops.maximum(self.parameters.sound_speed - sound_speed_offset, 1e-6)
        else:
            sound_speed = self.parameters.sound_speed

        wavelength = sound_speed / self.parameters.center_frequency

        positions = self.positions + (
            out["position_offset"] * wavelength
            if "position_offset" in out
            else ops.zeros_like(self.positions))

        return positions, sound_speed

    def forward(self, image, optvars, seed, freq_gaussian_probs=False, **kwargs):
        assert isinstance(optvars, dict), "optvars should be a dict"
        positions, sound_speed = self.reparameterize_optvars(optvars)

        assert len(image.shape) == 4, "Image should have shape (B, H, W, C)"
        image = ops.image.resize(image, self.shape, interpolation="bilinear")
        n_frames = image.shape[0]
        magnitudes = self.image_to_magnitudes(image, n_frames)

        seed_el, seed_freq, seed_tx = split_seed(seed, 3)
        
        tx_random = keras.random.uniform(shape=(len(self.beams),), seed=seed_tx)
        tx_indices = ops.argsort(tx_random)[:self.n_tx_samples]

        ax_random = keras.random.uniform(shape=(self.parameters.n_ax,), seed=seed_freq)
        ax_indices = ops.argsort(ax_random)[:self.n_ax_samples]

        el_random = keras.random.uniform(shape=(self.parameters.n_el,), seed=seed_el)
        el_indices = ops.argsort(el_random)[:self.n_el_samples]

        tx_indices = ops.sort(tx_indices)
        ax_indices = ops.sort(ax_indices)
        el_indices = ops.sort(el_indices)

        # Compute scatterer padding for chunking
        n_scat = positions.shape[0]
        chunk_size = self.scatterer_chunk_size
        n_scat_padded = n_scat + (chunk_size - (n_scat % chunk_size)) % chunk_size
        n_chunks = n_scat_padded // chunk_size

        #pad and reshape positions, cell sizes and magnitudes for chunking
        positions_padded = ops.pad(positions, ((0, n_scat_padded - n_scat), (0, 0)))
        position_chunks = ops.reshape(positions_padded, (n_chunks, chunk_size, 3))
        magnitudes_padded = ops.pad(magnitudes, ((0, 0), (0, n_scat_padded - n_scat)))

        def simulate_chunk(pos_chunk, mag_chunk, cell_chunk):
            """Simulate RF for a single scatterer chunk."""
            return simulate_partial_rf_data(
                scatterer_positions=pos_chunk,
                scatterer_magnitudes=mag_chunk,
                el_indices=el_indices,
                ax_indices=ax_indices,
                tx_indices=tx_indices,
                probe_geometry=self.parameters.probe_geometry,
                apply_lens_correction=self.parameters.apply_lens_correction,
                sound_speed=sound_speed,
                lens_sound_speed=self.parameters.lens_sound_speed,
                lens_thickness=self.parameters.lens_thickness,
                n_ax=self.parameters.n_ax - self.n_ax_min,
                center_frequency=self.parameters.center_frequency,
                sampling_frequency=self.parameters.sampling_frequency,
                t0_delays=self.parameters.t0_delays,
                initial_times=self.parameters.initial_times + self.initial_time_offset,
                element_width=self.parameters.element_width,
                attenuation_coef=self.parameters.attenuation_coef,
                tx_apodizations=self.parameters.tx_apodizations,
            )

        @jax.checkpoint
        def accumulate_chunks(rf_accumulated, chunk):
            pos_chunk, mag_chunk = chunk
            rf_accumulated = rf_accumulated + simulate_chunk(pos_chunk, mag_chunk)
            return rf_accumulated, None

        @jax.checkpoint
        def process_frame(carry, frame_data):
            magnitude_chunks = ops.reshape(frame_data, (n_chunks, chunk_size))
            rf_init = ops.zeros((self.n_tx_samples, self.n_freq_samples, self.n_el_samples, 1), dtype='complex64')
            rf_accumulated, _ = ops.scan(accumulate_chunks, rf_init, (position_chunks, magnitude_chunks))
            return carry, rf_accumulated

        _, rf_data = ops.scan(process_frame, None, magnitudes_padded)
        return rf_data, tx_indices, ax_indices, el_indices

    def __str__(self):
        return f"Simulator_Total with {self.n_tx_samples} tx samples, {self.n_freq_samples} freq samples, and {self.n_el_samples} el samples. Optimizing position_offset and sound_speed_offset."

    def transpose(self, data, *args, **kwargs):
        raise NotImplementedError("Transpose not implemented for Simulator_Time operator.")


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
             n_random_beams = 0,
             grid_type = 'scan',
             ax_min = 0):
        super().__init__()
        self.parameters = self.validate_parameters(scan)
        self.shape = scan.grid.shape[:2]
        self.n_tx_samples = n_tx_samples
        self.n_freqs = 2**int(ops.ceil(ops.log2(self.parameters.n_ax))) // 2 + 1
        self.n_freq_samples = n_freq_samples
        self.n_el_samples = n_el_samples
        self.scatterer_chunk_size = scatterer_chunk_size
        
        self.base_sound_speed = jnp.asarray(self.parameters.sound_speed, dtype=jnp.float32)
        self.base_initial_times = jnp.asarray(self.parameters.initial_times, dtype=jnp.float32)+ax_min/self.parameters.sampling_frequency
        self.base_attenuation_coef = jnp.asarray(self.parameters.attenuation_coef, dtype=jnp.float32)
        
        self.beams = self.select_beams(n_equidistant_beams, n_random_beams)
        self.grid_type = grid_type
        self.ax_min = ax_min
        if grid_type == 'scan':
            self.base_positions = self.parameters.flatgrid
        elif grid_type == "cone":
            self.cone_distance_to_apex = scan.distance_to_apex
            self.base_positions = self.cone_grid(scan)

    def validate_parameters(self, scan):
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
    
    def cone_grid(self, scan):
        """
        Generate a cone-shaped grid for beamforming. but the density of points is equal in the whole domain. not concentrated at the top
        """
        #Generate grid of points, between xlims and from apex to z_max
        x_min, x_max = self.xlims(scan)
        z_min, z_max = scan.zlims
        max_angle = np.max(np.abs(scan.polar_limits))
        t = np.tan(max_angle)
        if np.isclose(t, 0.0):
            t = 0.0
        distance_to_apex = scan.distance_to_apex

        if scan.pixels_per_wavelength is not None:
            n_pix_x = (x_max - x_min)/scan.wavelength*scan.pixels_per_wavelength
            n_pix_z = (z_max - z_min+distance_to_apex)/scan.wavelength*scan.pixels_per_wavelength
            x = np.linspace(x_min, x_max, int(n_pix_x))
            z = np.linspace(z_min, z_max+distance_to_apex, int(n_pix_z))
        else:
            x = np.linspace(x_min, x_max, scan.grid_size_x)
            z = np.linspace(z_min, z_max+distance_to_apex, scan.grid_size_z)
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)
        positions = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        cone_mask = np.abs(X) <= Z * t
        positions = positions[cone_mask.reshape(-1)]

        r_mask = np.sqrt(positions[...,0]**2 + positions[...,2]**2) <= (z_max+distance_to_apex)
        positions = positions[r_mask.reshape(-1)]

        positions[...,2] -= distance_to_apex
        positions = positions[positions[...,2] >= 0]
        
        return ops.convert_to_tensor(positions, dtype="float32")

    def sample_image_on_positions(self, image, positions, fill_value=0.0, order=1):
        """Sample a polar/frustum image at arbitrary cone-grid positions."""
        if len(image.shape) != 4:
            raise ValueError(f"Image should have shape (B, H, W, C). Got {image.shape}.")
        if image.shape[-1] != 1:
            raise ValueError(
                f"Cone-grid sampling expects a single-channel image, got {image.shape[-1]} channels."
            )

        image = ops.squeeze(image, axis=-1)
        image_shape = image.shape
        rho_range = getattr(self.parameters, "rho_range", None)
        if rho_range is None:
            rho_range = (self.parameters.zlims[0], self.parameters.zlims[1] + self.cone_distance_to_apex)
        theta_range = getattr(self.parameters, "theta_range", None)
        if theta_range is None:
            theta_range = self.parameters.polar_limits

        rho_min, rho_max = rho_range
        theta_min, theta_max = theta_range

        x = positions[:, 0]
        z = positions[:, 2] + self.cone_distance_to_apex
        rho, theta = frustum_convert_xz2rt(x, z, (theta_min, theta_max))

        valid = (
            (rho >= rho_min)
            & (rho <= rho_max)
            & (theta >= theta_min)
            & (theta <= theta_max)
        )

        rho_idx = (rho - rho_min) / (rho_max - rho_min) * (image_shape[-2] - 1)
        theta_idx = (theta - theta_min) / (theta_max - theta_min) * (image_shape[-1] - 1)

        rho_idx = ops.where(valid, rho_idx, -1.0)
        theta_idx = ops.where(valid, theta_idx, -1.0)
        coordinates = ops.stack([rho_idx, theta_idx], axis=0)

        def _sample_single(single_image):
            return map_coordinates(
                single_image,
                coordinates,
                order=order,
                fill_mode="constant",
                fill_value=fill_value,
            )

        return ops.vectorized_map(_sample_single, image)

    def xlims(scan):
        #To correct error in zea
        radius = max(scan.zlims+scan.distance_to_apex)
        xlims_polar = (
            radius * np.cos(-np.pi / 2 + scan.polar_limits[0]),
            radius * np.cos(-np.pi / 2 + scan.polar_limits[1]),
        )
        xlims_plane = (min(scan.probe_geometry[:, 0]), max(scan.probe_geometry[:, 0]))
        xlims = (
            min(xlims_polar[0], xlims_plane[0]),
            max(xlims_polar[1], xlims_plane[1]),
        )
        return xlims

    def select_beams(self, n_equidistant, n_random):
        if n_equidistant == "all" or n_random == "all":
            return ops.arange(self.parameters.n_tx, dtype="int32")
        
        if n_equidistant == 0 and n_random == 0:
            raise ValueError("At least one of n_equidistant_beams or n_random_beams must be greater than 0.")
        if n_equidistant > 0 and n_random > 0:
            raise ValueError("n_equidistant_beams and n_random_beams cannot both be greater than 0. Please choose one sampling strategy.")
        if n_equidistant > 0:
            #choose n equidistant transmits from the scan
            beams = ops.linspace(0, self.parameters.n_tx-1, n_equidistant, dtype="int32")
        if n_random > 0:
            #choose n random transmits from the scan
            beams = ops.random.shuffle(ops.arange(self.parameters.n_tx, dtype="int32"))[:n_random]
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
            # Positive offset should increase sound speed, matching caller parameterization.
            sound_speed = ops.cast(self.base_sound_speed + out["sound_speed_offset"], "float32")
        else:
            sound_speed = self.base_sound_speed
        sound_speed = ops.maximum(sound_speed, 1e-6)
        
        wavelength = sound_speed / self.parameters.center_frequency
        
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
            element_gains = ops.ones((self.parameters.n_el,), dtype="float32") + out["element_gains"] * 1e-3
        else:
            element_gains = ops.ones((self.parameters.n_el,), dtype="float32")
        
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
        if self.grid_type == "cone":
            magnitudes = self.sample_image_on_positions(image, positions)
        else:
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
        # Gumbel top-k sampling: pick largest scores, not smallest.
        freq_indices = ops.argsort(freq_random)[-self.n_freq_samples:]

        el_random = keras.random.uniform(shape=(self.parameters.n_el,), seed=seed_el)
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
                probe_geometry=self.parameters.probe_geometry,
                apply_lens_correction=self.parameters.apply_lens_correction,
                sound_speed=sound_speed,
                lens_sound_speed=self.parameters.lens_sound_speed,
                lens_thickness=self.parameters.lens_thickness,
                n_ax=self.parameters.n_ax,
                center_frequency=self.parameters.center_frequency,
                sampling_frequency=self.parameters.sampling_frequency,
                t0_delays=self.parameters.t0_delays,
                initial_times=initial_times,
                element_width=self.parameters.element_width,
                attenuation_coef=attenuation_coef,
                tx_apodizations=self.parameters.tx_apodizations,
                tgc_gain_curve=self.parameters.tgc_gain_curve,
            )

        @jax.checkpoint
        def accumulate_chunks(rf_accumulated, chunk):
            scat_chunk, amp_chunk = chunk
            rf_accumulated = rf_accumulated + simulate_chunk(scat_chunk, amp_chunk)
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