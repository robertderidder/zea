import numpy as np


def fish():
    """Returns a scatterer phantom for ultrasound simulation tests.

    Returns:
        ndarray: The scatterer positions of shape (104, 3).
    """
    # The size is the height of the fish
    size = 11e-3
    z_offset = 2.0 * size

    # See https://en.wikipedia.org/wiki/Fish_curve
    def fish_curve(t, size=1):
        x = size * (np.cos(t) - np.sin(t) ** 2 / np.sqrt(2))
        y = size * np.cos(t) * np.sin(t)
        return x, y

    scat_x, scat_z = fish_curve(np.linspace(0, 2 * np.pi, 100), size=size)

    scat_x = np.concatenate(
        [
            scat_x,
            np.array([size * 0.7]),
            np.array([size * 1.1]),
            np.array([size * 1.4]),
            np.array([size * 1.2]),
        ]
    )
    scat_y = np.zeros_like(scat_x)
    scat_z = np.concatenate(
        [
            scat_z,
            np.array([-size * 0.1]),
            np.array([-size * 0.25]),
            np.array([-size * 0.6]),
            np.array([-size * 1.0]),
        ]
    )

    scat_z += z_offset
    scat = np.stack([scat_x, scat_y, scat_z], axis=1)
    return scat


def rose(k=3, num_scatterers=80):
    """Generates scatterer positions based on the Rose curve.

    Args:
        k (int): The frequency parameter of the rose curve.
        num_scatterers (int): The number of scatterers to generate along the curve.

    Returns:
        ndarray: The scatterer positions of shape (num_scatterers, 3).
    """
    # https://en.wikipedia.org/wiki/Rose_(mathematics)
    size = 11e-3
    z_offset = 2.0 * size

    def rose_curve(theta, k):
        r = size * np.cos(k * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    scat_x, scat_z = rose_curve(theta=np.linspace(0, 2 * np.pi, num_scatterers), k=k)
    scat_z += z_offset
    scat_y = np.zeros_like(scat_x)
    scat = np.stack([scat_x, scat_y, scat_z], axis=1)
    return scat


def fibonacci(alpha=0.5, num_scatterers=100):
    """Generates scatterer positions based on the Fibonacci sphere algorithm.

    Args:
        num_scatterers (int): Number of scatterer points to generate.
        alpha (float): Scaling factor for the z-coordinate.

    Returns:
        ndarray: The scatterer positions of shape (num_scatterers, 3).
    """
    size = 11e-3
    z_offset = 2.0 * size

    phi = (1 + 5**0.5) / 2
    golden_angle = 2 * np.pi * (1 - 1 / phi)

    n = np.arange(num_scatterers)
    r = size * np.sqrt((n + alpha) / num_scatterers)
    theta = n * golden_angle

    scat_x = r * np.cos(theta)
    scat_z = r * np.sin(theta) + z_offset
    scat_y = np.zeros_like(scat_x)

    scat = np.stack([scat_x, scat_y, scat_z], axis=1)
    return scat


def lissajous(a=3, b=2, delta=np.pi / 2, num_scatterers=200):
    """Generates scatterer positions based on a Lissajous curve.

    Args:
        a (int): Frequency parameter for the x-coordinate.
        b (int): Frequency parameter for the z-coordinate.
        num_scatterers (int): Number of scatterer points to generate.

    Returns:
        ndarray: The scatterer positions of shape (num_scatterers, 3).
    """
    size = 11e-3
    z_offset = 2.0 * size

    t = np.linspace(0, 2 * np.pi, num_scatterers, endpoint=False)

    scat_x = size * np.sin(a * t + delta)
    scat_z = size * np.sin(b * t) + z_offset
    scat_y = np.zeros_like(scat_x)
    scat = np.stack([scat_x, scat_y, scat_z], axis=1)
    return scat


def golden_ratio(num_scatterers=100):
    """Generates scatterer positions based on the golden ratio spiral.

    Args:
        num_scatterers (int): Number of scatterer points to generate.

    Returns:
        ndarray: The scatterer positions of shape (num_scatterers, 3).
    """
    size = 11e-3
    z_offset = 2.0 * size

    phi = (1 + 5**0.5) / 2  # Golden ratio
    theta = np.linspace(0, 4 * np.pi, num_scatterers)
    r = size * (theta / (4 * np.pi))

    scat_x = r * np.cos(theta * phi)
    scat_z = r * np.sin(theta * phi) + z_offset
    scat_y = np.zeros_like(scat_x)
    scat = np.stack([scat_x, scat_y, scat_z], axis=1)
    return scat

def circle(n_points=50, radius=0.01, z_pos=22e-3):
    """Generates a circular phantom of scatterers in the x-y plane.

    Args:
        radius (float): The radius of the circle in meters.
        n_points (int): The number of scatterer points to generate.
        z_pos (float): The z-coordinate for all scatterers.

    Returns:
        ndarray: The scatterer positions of shape (n_points, 3).
    """
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(angles)
    z = radius * np.sin(angles)
    y = np.zeros_like(x)
    z = z+z_pos
    return np.stack([x, y, z], axis=1)

def grid(n_x=5, n_z=5, x_range=(-10e-3, 10e-3), z_range=(15e-3, 30e-3)):
    """Returns a grid of point scatterers."""
    x = np.linspace(x_range[0], x_range[1], n_x)
    z = np.linspace(z_range[0], z_range[1], n_z)
    X, Z = np.meshgrid(x, z)
    scat_x = X.flatten()
    scat_y = np.zeros_like(scat_x)
    scat_z = Z.flatten()
    return np.stack([scat_x, scat_y, scat_z], axis=1)