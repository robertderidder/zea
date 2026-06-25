"""Frequency domain ultrasound simulator.

The simulator works in the frequency domain (RFFT domain) and simulates RF data as a superposition
of scatterer responses. Every scatterer has a location and a magnitude.

To use it in your code, simply call the :func:`simulate_rf` function with the desired
transmit scheme parameters and scatterers. To simulate a sequence of multiple frames,
you can call :func:`simulate_rf` repeatedly with different scatterer positions and magnitudes
and then stack the results.

Example usage
^^^^^^^^^^^^^

A simple example of simulating RF data with a single scatterer at the center of the probe. For a
more in depth example see the notebook: :doc:`../notebooks/data/zea_simulation_example`.

.. doctest::

    >>> from zea.simulator import simulate_rf
    >>> import numpy as np

    >>> raw_data = simulate_rf(
    ...     scatterer_positions=np.array([[0, 0, 20e-3]]),
    ...     scatterer_magnitudes=np.array([1.0]),
    ...     probe_geometry=np.stack(
    ...         [np.linspace(-20e-3, 20e-3, 64), np.zeros(64), np.zeros(64)], axis=-1
    ...     ),
    ...     apply_lens_correction=True,
    ...     lens_thickness=1e-3,
    ...     lens_sound_speed=1000,
    ...     sound_speed=1540,
    ...     n_ax=1024,
    ...     center_frequency=5e6,
    ...     sampling_frequency=20e6,
    ...     t0_delays=np.zeros((1, 64)),
    ...     initial_times=np.zeros(1),
    ...     element_width=0.2e-3,
    ...     attenuation_coef=0.5,
    ...     tx_apodizations=np.ones((1, 64)),
    ... )

"""

import numpy as np
from keras import ops

from zea.beamform.lens_correction import compute_lens_corrected_travel_times
from zea.func.tensor import vmap

def simulate_rf(
    scatterer_positions,
    scatterer_magnitudes,
    el_indices,
    freq_indices,
    tx_indices,
    probe_geometry,
    apply_lens_correction,
    lens_thickness,
    lens_sound_speed,
    sound_speed,
    n_ax,
    center_frequency,
    sampling_frequency,
    t0_delays,
    initial_times,
    element_width,
    attenuation_coef,
    tx_apodizations,
    scatterer_cell_size=None,
    waveform=None,
    waveform_sampling_frequency=None,
):
    """
    Simulates a select number of rf_data points in the frequency domain for a given set of scatterers and probe geometry.


    Args:
        scatterer_positions (array-like): The positions of the scatterers [m] of shape (n_scat, 3).
        scatterer_magnitudes (array-like): The magnitudes of the scatterers of shape (n_scat,).
        probe_geometry (array-like): The geometry of the probe [m] of shape (n_el, 3).
        apply_lens_correction (bool): Whether to apply lens correction.
        lens_thickness (float): The thickness of the lens [m].
        lens_sound_speed (float): The speed of sound in the lens [m/s].
        sound_speed (float): The speed of sound in the medium [m/s].
        n_ax (int): The number of samples in the RF data.
        center_frequency (float): The center frequency of the transmit pulse [Hz].
        sampling_frequency (float): The sampling frequency of the RF data [Hz].
        t0_delays (array-like): The delays of the transmitting elements [s] of shape (n_tx, n_el).
        initial_times (array-like): The initial times of the transmitting elements [s] of
            shape (n_tx,).
        element_width (float): The width of the elements [m].
        attenuation_coef (float): The attenuation coefficient [dB/cm/MHz].
        tx_apodizations (array-like): The apodizations of the transmitting elements of
            shape (n_tx, n_el).
        scatterer_cell_size (array-like, optional): The (dz, dx) size [m] of the Riemann-sum
            cell each scatterer represents, of shape (n_scat, 2). ``dz`` is the radial/axial
            extent and ``dx`` the lateral extent. When given, each scatterer is treated as a
            finite-size patch rather than a Dirac point: a `sinc`-shaped directivity term
            (sized by ``dx``, mirroring the probe-element ``directivity``) suppresses lateral
            grating lobes, and a `sinc`-shaped frequency roll-off (sized by ``dz``) accounts
            for the spread of round-trip path lengths across the cell's depth. Both terms tend
            to 1 as the cell shrinks, recovering the point-scatterer model. If ``None``
            (default), scatterers are treated as ideal points (previous behaviour).
        waveform (array-like, optional): A 1D measured transmit waveform of shape
            (n_samp,) (e.g. ``scan.waveforms_two_way[0]``). When given, the simulator
            multiplies each scatterer's response by this waveform's spectrum instead of
            the default analytic Hann-windowed pulse, making the simulated pulse shape
            match the real transducer. Requires ``waveform_sampling_frequency``. If
            ``None`` (default), the analytic pulse is used (previous behaviour).
        waveform_sampling_frequency (float, optional): The sampling rate [Hz] of
            ``waveform`` (e.g. 250e6 for Verasonics waveforms). Only used when
            ``waveform`` is given.

    Returns:
        fourier transform of the RF data. With 1D ``scatterer_magnitudes`` of shape
        (n_scat,) the shape is (n_tx, n_freq, n_el, 1); with a batched (n_frames, n_scat)
        the leading frame axis is preserved: (n_frames, n_tx, n_freq, n_el, 1).
    """

    probe_geometry_tx = probe_geometry
    probe_geometry_rx = ops.take(probe_geometry, el_indices, axis=0)

    t0_delays = ops.take(t0_delays, tx_indices, axis=0)

    initial_times = ops.take(initial_times, tx_indices, axis=0)

    tx_apodizations = ops.take(tx_apodizations, tx_indices, axis=0)

    if waveform is None:
        pulse_spectrum_fn = get_pulse_spectrum_fn(center_frequency, n_period=4)
    else:
        pulse_spectrum_fn = get_measured_waveform_spectrum_fn(
            waveform, waveform_sampling_frequency
        )

    if not apply_lens_correction:
        dist_tx = ops.linalg.norm(probe_geometry_tx[None] - scatterer_positions[:, None], axis=-1)
        dist_rx = ops.linalg.norm(probe_geometry_rx[None] - scatterer_positions[:, None], axis=-1)
    else:
        dist_tx = (
            compute_lens_corrected_travel_times(
                probe_geometry_tx,
                scatterer_positions,
                lens_thickness=lens_thickness,
                c_lens=lens_sound_speed,
                c_medium=sound_speed,
                n_iter=3,
            )
            * sound_speed
        )
        dist_rx = (
            compute_lens_corrected_travel_times(
                probe_geometry_rx,
                scatterer_positions,
                lens_thickness=lens_thickness,
                c_lens=lens_sound_speed,
                c_medium=sound_speed,
                n_iter=3,
            )
            * sound_speed
        )

    n_ax_rounded = _round_up_to_power_of_two(int(n_ax)).astype("float32")

    freqs = ops.arange(n_ax_rounded // 2 + 1, dtype="float32") / n_ax_rounded * sampling_frequency
    freqs = ops.take(freqs, freq_indices)

    waveform_spectrum = pulse_spectrum_fn(freqs)
    # [n_scat, n_txel, n_rxel]
    dist_total = dist_tx[:, :, None] + dist_rx[:, None, :]

    scat_pos_relative_to_probe_tx = scatterer_positions[:, None] - probe_geometry_tx[None]
    scat_pos_relative_to_probe_rx = scatterer_positions[:, None] - probe_geometry_rx[None]

    # Compute 3D directivity terms that are independent of transmit index.
    theta_tx = ops.arctan2(
        scat_pos_relative_to_probe_tx[:, :, 0], scat_pos_relative_to_probe_tx[:, :, 2]
    )
    phi_tx = ops.arctan2(
        scat_pos_relative_to_probe_tx[:, :, 1], scat_pos_relative_to_probe_tx[:, :, 2]
    )
    theta_rx = ops.arctan2(
        scat_pos_relative_to_probe_rx[:, :, 0], scat_pos_relative_to_probe_rx[:, :, 2]
    )
    phi_rx = ops.arctan2(
        scat_pos_relative_to_probe_rx[:, :, 1], scat_pos_relative_to_probe_rx[:, :, 2]
    )

    directivity_tx = directivity(
        freqs[None, None, None],
        theta_tx[..., None, None],
        element_width,
        sound_speed,
    ) * directivity(
        freqs[None, None, None],
        phi_tx[..., None, None],
        element_width,
        sound_speed,
    )
    directivity_rx = directivity(
        freqs[None, None, None],
        theta_rx[:, None, :, None],
        element_width,
        sound_speed,
    ) * directivity(
        freqs[None, None, None],
        phi_rx[:, None, :, None],
        element_width,
        sound_speed,
    )

    attenuation = attenuate(
        freqs[None, None, None],
        attenuation_coef=attenuation_coef,
        dist=dist_total[..., None],
    )

    spread_atten = spread(dist_total[..., None])

    if scatterer_cell_size is not None:
        dz_cell = scatterer_cell_size[:, 0]
        dx_cell = scatterer_cell_size[:, 1]

        # Finite lateral extent: suppress grating lobes the same way the
        # finite-width probe elements do, but sized by the scatterer cell.
        cell_directivity_tx = directivity(
            freqs[None, None, None],
            theta_tx[..., None, None],
            dx_cell[:, None, None, None],
            sound_speed,
        )
        cell_directivity_rx = directivity(
            freqs[None, None, None],
            theta_rx[:, None, :, None],
            dx_cell[:, None, None, None],
            sound_speed,
        )

        # NOTE: an axial sinc term (sinc(f * dz * (cos_tx + cos_rx) / c)) was
        # previously included here to model finite radial extent. It was removed
        # because for typical grid spacings (dz ≈ 0.5–1.5 wavelengths), the
        # sinc argument at center frequency falls past its first zero crossing
        # (arg > 1), causing the forward model to produce near-zero signal with
        # wrong sign — effectively killing the DPS gradient.
        cell_response = cell_directivity_tx * cell_directivity_rx
    else:
        cell_response = 1.0

    def _kernel_single_tx(t0_delay_tx, initial_time_tx, tx_apodization_tx):
        # Per-transmit RF response kernel with the scatterer magnitudes factored OUT. The
        # forward model is linear in `scatterer_magnitudes`, so we build the (constant w.r.t.
        # the optimization variable) kernel here and apply the magnitudes via a single
        # contraction below. This lets reverse-mode AD store only this small post-`tx_el`-sum
        # kernel — not the huge (n_scat, n_tx_el, n_rx, n_freq) intermediate — so the
        # likelihood gradient costs one kernel build instead of a full forward recompute.
        tau_total = (dist_total / sound_speed) + t0_delay_tx[None, :, None] - initial_time_tx

        kernel = (
            waveform_spectrum[None, None, None]
            * delay2(
                freqs[None, None, None],
                tau_total[..., None],
                n_fft=n_ax_rounded,
                sampling_frequency=sampling_frequency,
            )
            * ops.cast(
                tx_apodization_tx[None, :, None, None]
                * directivity_tx
                * directivity_rx
                * attenuation
                * spread_atten
                * cell_response,
                "complex64",
            )
        )

        # Sum over transmitting elements only; keep the scatterer axis for the magnitude
        # contraction outside. -> (n_scat, n_rx, n_freq)
        return ops.sum(kernel, axis=1)

    # (n_tx, n_scat, n_rx, n_freq)
    kernel = vmap(_kernel_single_tx)(t0_delays, initial_times, tx_apodizations)

    # Linear contraction with the (area-weighted) scatterer magnitudes. `scatterer_magnitudes`
    # may carry a leading frame batch axis: (n_scat,) -> (n_tx, n_freq, n_rx, 1);
    # (n_frames, n_scat) -> (n_frames, n_tx, n_freq, n_rx, 1). Complex-safe (per-scatterer
    # phase is allowed). Summing over n_scat here is mathematically identical to the previous
    # combined (scat, tx_el) sum — only the reduction order differs.
    mag = ops.cast(scatterer_magnitudes, "complex64")
    if len(mag.shape) == 1:
        rf_data = ops.einsum("s,tsrf->trf", mag, kernel)
        rf_data = ops.transpose(rf_data, (0, 2, 1))[..., None]
    else:
        rf_data = ops.einsum("ns,tsrf->ntrf", mag, kernel)
        rf_data = ops.transpose(rf_data, (0, 1, 3, 2))[..., None]
    return rf_data

def directivity(f, theta, element_width, sound_speed, rigid_baffle=True):
    """Computes the directivity of a single element.

    Args:
        f (array-like): The input frequencies [Hz].
        theta (array-like): The angles [rad].
        element_width (float): The width of the element [m].
        sound_speed (float): The speed of sound [m/s].
        rigid_baffle (bool): Whether the element is mounted on a rigid baffle,
            impacting the directivity.

    Returns:
        array-like: The directivity of the element.
    """

    # Use f / c form instead of c / f to avoid undefined gradients at f=0 (DC bin).
    safe_sound_speed = ops.maximum(sound_speed, 1e-6)
    argument = (element_width * f / safe_sound_speed) * ops.sin(theta)
    response = sinc(argument)
    if not rigid_baffle:
        response *= ops.cos(theta)
    return response


def delay2(f, tau, n_fft, sampling_frequency):
    """
    Applies a delay in the frequency domain without phase wrapping.

    Args:
        f (array-like): The input frequencies.
        tau (float): The delay to apply.
        n_fft (int): The number of samples in the FFT.
        sampling_frequency (float): The sampling frequency.

    Returns:
        array-like: The spectrum of the delay.
    """
    arg = ops.array(-1j, dtype="complex64") * ops.cast(2 * np.pi * tau * f, "complex64")
    return ops.where(
        tau < n_fft / sampling_frequency,
        ops.exp(arg),
        ops.array(0.0, dtype="complex64"),
    )


def attenuate(f, attenuation_coef, dist):
    """
    Applies attenuation to the signal in the frequency domain.

    Args:
        f (array-like): The input frequencies.
        attenuation_coef (float): The attenuation coefficient in dB/cm/MHz.
        dist (float): The distance the signal has traveled.

    Returns:
        array-like: The spectrum of the attenuation.
    """
    return ops.exp(-ops.log(10) * attenuation_coef / 20 * dist * 100 * ops.abs(f) * 1e-6)


def spread(dist, mindist=1e-4):
    """Function modeling geometric spreading of the wavefront.

    Args:
        dist (array-like): The distance the wave has traveled.
        mindist (float): The minimum distance to prevent division by zero.

    Returns:
        array-like: The geometric spreading factor of same shape as `dist`.
    """
    dist = ops.clip(dist, mindist, float("inf"))
    return mindist / dist


def hann_fd(f, width):
    """The fourier transform of a hann window in the time domain with given width."""
    denom = 1.0 - (f * width) ** 2
    num = 0.5 * sinc(f * width)
    result = num / denom
    result = ops.where(ops.abs(result) > 1.1, 0.25, result)
    return ops.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.25)


def hann_unnormalized(x, width):
    """Hann window function that is 1 at the peak. This means that the integral of the
    window function is not necessarily 1.

    Args:
        x (array-like): The input values.
        width (float): The width of the window. This is the total width from -x to x. The
            window will be nonzero in the range [-width/2, width/2].

    Returns:
        hann_vals (array-like): The values of the Hann window function.
    """
    return ops.where(ops.abs(x) < width / 2, ops.cos(np.pi * x / width) ** 2, 0)


def get_pulse_spectrum_fn(center_frequency, n_period=3.0):
    """Computes the spectrum of a sine that is windowed with a Hann window.

    Args:
        center_frequency (float): The center frequency of the transmit pulse.
        n_period (float): The number of periods to include in the pulse.

    Returns:
        spectrum_fn (callable): A function that computes the spectrum of the pulse
        for the input frequencies in Hz.
    """
    period = n_period / center_frequency

    def spectrum_fn(f):
        return ops.array(1 / 2, "complex64") * ops.cast(
            (hann_fd(f - center_frequency, period) + hann_fd(f + center_frequency, period)),
            "complex64",
        )

    return spectrum_fn


def get_measured_waveform_spectrum_fn(waveform, waveform_sampling_frequency):
    """Computes the spectrum of a *measured* transmit waveform.

    Evaluates the discrete-time Fourier transform (DTFT) of the time-domain
    waveform directly at the requested frequencies::

        W(f) = sum_n w[n] * exp(-j 2 pi f n / fs_wave)

    Evaluating the DTFT directly (rather than an FFT followed by interpolation)
    lets us sample the spectrum at the simulator's arbitrary, per-step-subsampled
    ``freqs`` and cleanly bridges the waveform's own sampling rate
    (``waveform_sampling_frequency``, e.g. 250 MHz for Verasonics) and the RF
    grid's sampling rate.

    The waveform is used as-is (indexed from sample 0), so its intrinsic latency
    is preserved through the phase of the spectrum. The result is normalized so
    its peak magnitude over the band is ~1, matching the scale of the analytic
    Hann pulse (:func:`get_pulse_spectrum_fn`) so the rest of the model
    (omega/eps/magnitude_range tuning) is unaffected.

    NOTE: only a single 1D waveform is supported. The current datasets use one
    shared transmit waveform for all transmits; per-transmit selection via
    ``tx_waveform_indices`` is a possible future extension.

    Args:
        waveform (array-like): 1D time-domain transmit waveform of shape (n_samp,).
        waveform_sampling_frequency (float): Sampling rate of ``waveform`` [Hz].

    Returns:
        spectrum_fn (callable): A function mapping frequencies [Hz] to the complex
        spectrum of the waveform.
    """
    assert waveform_sampling_frequency is not None, (
        "waveform_sampling_frequency must be given when using a measured waveform."
    )
    waveform_np = np.asarray(waveform, dtype="float32").reshape(-1)

    # Fixed normalization constant: the peak magnitude of the full-resolution
    # spectrum. Computed once (not per-call) so the scale is stable regardless of
    # which frequency subset is sampled in a given forward pass.
    norm = float(np.max(np.abs(np.fft.rfft(waveform_np))))
    norm = norm if norm > 0 else 1.0

    waveform_t = ops.convert_to_tensor(waveform_np, dtype="float32")
    sample_indices = ops.arange(waveform_np.shape[0], dtype="float32")

    def spectrum_fn(f):
        # arg has shape (..., n_freq, n_samp); exp(j*arg) is the DTFT basis.
        arg = -2 * np.pi * f[..., None] * sample_indices / waveform_sampling_frequency
        basis = ops.cast(ops.cos(arg), "complex64") + ops.cast(
            1j, "complex64"
        ) * ops.cast(ops.sin(arg), "complex64")
        spectrum = ops.sum(ops.cast(waveform_t, "complex64") * basis, axis=-1)
        return spectrum / ops.cast(norm, "complex64")

    return spectrum_fn


def get_transducer_bandwidth_fn(probe_center_frequency, bandwidth):
    """Computes the spectrum of a probe with a center frequency and bandwidth.

    Args:
        probe_center_frequency (float): The center frequency of the probe.
        bandwidth (float): The bandwidth of the probe.

    Returns
        spectrum_fn (callable): A function that computes the spectrum of the pulse for
        the input frequencies in Hz.
    """

    def bandwidth_fn(f):
        return hann_unnormalized(ops.abs(f) - probe_center_frequency, bandwidth)

    return bandwidth_fn


def sinc(x):
    """The normalized sinc function with a small offset to prevent division by zero."""
    x = ops.abs(np.pi * x) + 1e-9
    return ops.sin(x) / x


def _round_up_to_power_of_two(x):
    """Rounds up to the next power of two."""
    return 2 ** np.ceil(np.log2(x))