from typing import Union
import jax
import jax.numpy as jnp
from jax import device_put, jit, vmap, lax
from functools import partial
from zea import log

def _compute_padding(n, chunk_size):
    """Compute padding needed to make n a multiple of chunk_size.
    
    Returns the number of elements to pad.
    """
    return (chunk_size - (n % chunk_size)) % chunk_size

def _get_vectorized_simulate_function(
    t0_delays,
    probe_geometry,
    element_angles,
    tx_apodization,
    initial_time,
    element_width_wl,
    sampling_frequency,
    sound_speed,
    attenuation_coefficient,
    waveform_function,
    wavefront_only=False,
    tx_angle_sensitivity=True,
    rx_angle_sensitivity=True,
):
    def _simulate_rf_sample(
        ax_index,
        element_index,
        scatterer_position,
        scatterer_amplitude,
    ):
        sampling_period = 1 / sampling_frequency
        rx_element_pos = probe_geometry[element_index]
        t_sample = ax_index * sampling_period
        t_rx = jnp.linalg.norm(scatterer_position - rx_element_pos) / sound_speed
        
        t_tx = (
            jnp.linalg.norm(probe_geometry - scatterer_position[None, :], axis=1)
            / sound_speed
            + t0_delays
        )

        delay = t_rx + t_tx

        if wavefront_only:
            delay = jnp.min(delay)

        if tx_angle_sensitivity:
            thetatx = jnp.arctan2(
                (scatterer_position[None, 0] - probe_geometry[:, 0]),
                (scatterer_position[None, 1] - probe_geometry[:, 1]),
            )
            thetatx -= element_angles
            angular_response_tx = jnp.sinc(element_width_wl * jnp.sin(thetatx)) * jnp.cos(thetatx)
        else:
            angular_response_tx = 1.0

        if rx_angle_sensitivity:
            theta = jnp.arctan2(
                (scatterer_position[0] - rx_element_pos[0]),
                (scatterer_position[1] - rx_element_pos[1]),
            )
            theta -= element_angles[element_index]
            angular_response_rx = jnp.sinc(element_width_wl * jnp.sin(theta)) * jnp.cos(theta)
        else:
            angular_response_rx = 1.0

        attenuation = jnp.where(
            attenuation_coefficient != 0.0,
            jnp.exp(-attenuation_coefficient * delay * sound_speed),
            1.0
        )

        response = (
            waveform_function(t_sample + initial_time - delay)
            * tx_apodization
            * scatterer_amplitude
            * angular_response_tx
            * angular_response_rx
            * attenuation
        )

        return jnp.sum(response)

    # Vectorize core logic
    vectorized_function = vmap(_simulate_rf_sample, in_axes=(None, None, 0, 0)) # scatterers
    vectorized_function = vmap(vectorized_function, in_axes=(None, 0, None, None)) # elements
    vectorized_function = vmap(vectorized_function, in_axes=(0, None, None, None)) # axial samples

    return vectorized_function

def simulate_rf_transmit(
    ax_indices: jnp.array,
    el_indices: jnp.array,
    scatterer_positions: jnp.array,
    scatterer_amplitudes: jnp.array,
    t0_delays: jnp.array,
    probe_geometry: jnp.array,
    element_angles: jnp.array,
    tx_apodization: jnp.array,
    initial_time: float,
    element_width_wl: float,
    sampling_frequency: Union[float, int],
    carrier_frequency: Union[float, int],
    sound_speed: Union[float, int] = 1540,
    attenuation_coefficient: Union[float, int] = 0.0,
    wavefront_only: bool = False,
    tx_angle_sensitivity: bool = True,
    rx_angle_sensitivity: bool = True,
    waveform_function=None,
    ax_chunk_size: int = 1024,
    scatterer_chunk_size: int = 1024,
):

    if waveform_function is None:
        tx_waveform = get_pulse(carrier_frequency, 3.0)
    else:
        tx_waveform = waveform_function

    if not wavefront_only:
        tx_waveform = vmap(tx_waveform)
    
    if wavefront_only is True:
        if tx_angle_sensitivity is True:
            tx_angle_sensitivity = False
            log.warning(
                "tx_angle_sensitivity is set to True while in wavefront_only mode. "
                "Changed to False."
            )
    # Convert attenuation (dB/MHz/cm -> Nepers/m)
    att_neper_m = attenuation_coefficient * jnp.log(10) / 20 * carrier_frequency * 1e-6 * 100

    simulation_function = _get_vectorized_simulate_function(
        t0_delays, 
        probe_geometry, 
        element_angles, 
        tx_apodization,
        initial_time, 
        element_width_wl, 
        sampling_frequency,
        sound_speed, 
        att_neper_m, 
        tx_waveform,
        wavefront_only, 
        tx_angle_sensitivity, 
        rx_angle_sensitivity
    )

    n_scat = scatterer_positions.shape[0]
    n_ax = len(ax_indices)
    n_el = len(el_indices)

    # Adaptive chunk sizes: don't pad if data already smaller than chunk size
    actual_ax_chunk_size = min(n_ax, ax_chunk_size)
    actual_scat_chunk_size = min(n_scat, scatterer_chunk_size)

    # Compute padding needed to make dimensions multiples of chunk sizes
    scat_pad = _compute_padding(n_scat, actual_scat_chunk_size)
    ax_pad = _compute_padding(n_ax, actual_ax_chunk_size)

    # Pad the arrays
    scatterer_positions_padded = jnp.pad(
        scatterer_positions,
        ((0, scat_pad), (0, 0)),  # pad rows (scatterers), not columns (coordinates)
        mode="constant",
        constant_values=0.0,
    )

    ax_indices_padded = jnp.pad(
        ax_indices,
        (0, ax_pad),  # pad the 1D array
        mode="edge",  # use edge values; we'll slice them out anyway
    )

    n_scat_padded = n_scat + scat_pad
    n_ax_padded = n_ax + ax_pad

    def frame_scan_fn(carry_unused, amplitudes_frame):
        # Pad amplitudes for this frame
        amplitudes_frame_padded = jnp.pad(
            amplitudes_frame,
            (0, scat_pad),
            mode="constant",
            constant_values=0.0,
        )
        
        def axial_scan_fn(carry_unused, ax_start):
            # Dynamic slice for axial indices (from padded array)
            curr_ax_indices = lax.dynamic_slice_in_dim(ax_indices_padded, ax_start, actual_ax_chunk_size)
            
            def scatterer_scan_fn(accumulated_rf, scat_start):
                # Dynamic slice for scatterers (from padded arrays)
                pos_chunk = lax.dynamic_slice_in_dim(scatterer_positions_padded, scat_start, actual_scat_chunk_size)
                amp_chunk = lax.dynamic_slice_in_dim(amplitudes_frame_padded, scat_start, actual_scat_chunk_size)
                
                # Compute and sum over scatterer dimension (axis 2)
                chunk_rf = simulation_function(curr_ax_indices, el_indices, pos_chunk, amp_chunk)
                return accumulated_rf + jnp.sum(chunk_rf, axis=2), None

            scat_starts = jnp.arange(0, n_scat_padded, actual_scat_chunk_size)
            init_rf = jnp.zeros((actual_ax_chunk_size, n_el))
            rf_ax_chunk, _ = lax.scan(scatterer_scan_fn, init_rf, scat_starts)
            
            return None, rf_ax_chunk

        ax_starts = jnp.arange(0, n_ax_padded, actual_ax_chunk_size)
        _, rf_frame_chunks = lax.scan(axial_scan_fn, None, ax_starts)
        
        # Reshape to (n_ax_padded, n_el), then slice out padding
        rf_frame_padded = rf_frame_chunks.reshape(-1, n_el)
        rf_frame = rf_frame_padded[:n_ax]  # slice out axial padding
        return None, rf_frame

    # Scan over frames (outermost)
    _, rf_data = lax.scan(frame_scan_fn, None, scatterer_amplitudes)
    return rf_data

@partial(jax.jit, static_argnames=(
    "wavefront_only",
))
def simulate_partial_rf_data(
    ax_indices: jnp.array,
    el_indices: jnp.array,
    tx_indices: jnp.array,
    scatterer_positions: jnp.array,
    scatterer_amplitudes: jnp.array,    
    t0_delays: jnp.array,
    probe_geometry: jnp.array,
    element_angles: jnp.array,
    tx_apodizations: jnp.array,
    initial_times: jnp.array,
    element_width_wl: float,
    sampling_frequency: Union[float, int],
    carrier_frequency: Union[float, int],
    sound_speed: Union[float, int] = 1540,
    attenuation_coefficient: Union[float, int] = 0.0,
    wavefront_only: bool = False,
    tx_angle_sensitivity: bool = True,
    rx_angle_sensitivity: bool = True,
    waveform_function: bool = None,
    ax_chunk_size: int = 1024,
    scatterer_chunk_size: int = 1024,
    verbose: bool = False
):
    """Simulates a subset of rf_data points. The axial, element and transmit indices have to be chosen in advance.
    Tis implementation is purely written in JAX so it can be jitted without running into OOM issues. 
        
        
        Parameters
        ----------
        ax_indices:
            The indices of the axial samples
        el_indices:
            The indices of the elements
        tx_indices : jnp.array
            The indices of the transmits to simulate. Shape (n_selected_tx,).
        scatterer_positions : jnp.array
            The scatterer positions of shape `(n_scat, 2)`.
        scatterer_amplitudes : jnp.array
            The scatterer amplitudes of shape `(n_frames, n_scat)`.
        t0_delays : jnp.array
            The t0_delays of shape `(n_tx, n_el)`. These are shifted such that the smallest value
            in t0_delays is 0.
        probe_geometry : jnp.array
            The probe geometry of shape `(n_el, 2)`.
        element_angles : jnp.array
            The element angles in radians of shape `(n_el,)`. Can be used to simulate curved
            arrays.
        tx_apodizations : jnp.array
            The transmit apodization of shape `(n_tx, n_el)`.
        initial_times : jnp.array
            The time instant of the first sample in seconds of shape `(n_tx,)`.
        element_width_wl : float
            The width of the elements in wavelengths of the center frequency.
        sampling_frequency : float
            The sampling frequency in Hz.
        carrier_frequency : float
            The center frequency of the transmit pulse in Hz.
        sound_speed : float
            The speed of sound in the medium.
        attenuation_coefficient : float
            The attenuation coefficient in dB/(MHz*cm)
        wavefront_only : bool
            Set to True to only compute the wavefront of the rf data. Otherwise the rf data
            is computed as the sum of the wavefronts from indivudual transmit elements.
        tx_angle_sensitivity : bool
            Set to True to include the angle dependent strength of the transducer elements
            in the response.
        rx_angle_sensitivity : bool
            Set to True to include the angle dependent
            strength of the transducer elements in the response.
        ax_chunk_size : int
            The number of axial samples to compute simultaneously.
        scatterer_chunk_size : int
            The number of scatterers to compute simultaneously.

        Returns
        -------
            jnp.array: The rf data of shape `(batch_size, n_selected_tx, n_ax, n_el, 1)`
        """

    if verbose:
        log.warning("Verbose is not supported in the JAX implementation of simulate_partial_rf_data. Ignoring verbose flag.")
    # Selection of transmit parameters
    selected_t0_delays = jnp.take(t0_delays, tx_indices, axis=0)
    selected_apodizations = jnp.take(tx_apodizations, tx_indices, axis=0)
    selected_times = jnp.take(initial_times, tx_indices, axis=0)

    def tx_scan_fn(carry_unused, tx_params):
        t0, apod, t_init = tx_params
        rf = simulate_rf_transmit(
            ax_indices, 
            el_indices, 
            scatterer_positions, 
            scatterer_amplitudes,
            t0, probe_geometry, 
            element_angles, 
            apod, 
            t_init,
            element_width_wl, 
            sampling_frequency, 
            carrier_frequency,
            sound_speed, 
            attenuation_coefficient, 
            wavefront_only,
            tx_angle_sensitivity, 
            rx_angle_sensitivity, 
            waveform_function,
            ax_chunk_size, 
            scatterer_chunk_size
        )
        return None, rf

    # Scan over selected transmits
    _, rf_all_tx = lax.scan(tx_scan_fn, None, (selected_t0_delays, selected_apodizations, selected_times))
    
    # Reorder to (n_frames, n_tx, n_ax, n_el, 1)
    rf_data = jnp.transpose(rf_all_tx, (1, 0, 2, 3))
    return rf_data[..., None]

def get_pulse(carrier_frequency, pulse_width_wl, chirp_rate=0, phase=0):
    def waveform_fn(t):
        pulse_width = 1/carrier_frequency * pulse_width_wl
        sigma = (0.5 * pulse_width) / jnp.sqrt(-jnp.log(0.1))
        t_shifted = t - pulse_width
        y = jnp.exp(-((t_shifted / sigma) ** 2))
        y *= jnp.sin(2 * jnp.pi * ((carrier_frequency + (chirp_rate * t_shifted)) * t_shifted) + phase)
        return y
    return waveform_fn