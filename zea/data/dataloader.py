"""H5 dataloader for loading images from zea datasets.

Example:
    .. code-block:: python

        from zea.data.dataloader import Dataloader

        loader = Dataloader(
            file_paths="/path/to/dataset",
            key="data/image",
            batch_size=16,
            image_range=(-60, 0),
            normalization_range=(0, 1),
            image_size=(256, 256),
            num_threads=16,
        )

        for batch in loader:
            # batch is a numpy array of shape (batch_size, 256, 256, 1)
            ...
"""

import os
import re
import threading
from itertools import product
from pathlib import Path
from typing import List, Tuple, Union
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import keras

from zea import log
from zea.data.datasets import Dataset, H5FileHandleCache, count_samples_per_directory
from zea.data.file import File
from zea.data.utils import json_dumps
from zea.io_lib import retry_on_io_error
from zea.utils import map_negative_indices

DEFAULT_NORMALIZATION_RANGE = (0, 1)
MAX_RETRY_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 0.1


def generate_h5_indices(
    file_paths: List[str],
    file_shapes: list,
    n_frames: int,
    frame_index_stride: int,
    key: str = "data/image",
    initial_frame_axis: int = 0,
    additional_axes_iter: List[int] | None = None,
    sort_files: bool = True,
    overlapping_blocks: bool = False,
    limit_n_frames: int | None = None,
):
    """Generate indices for h5 files.

    Generates a list of indices to extract images from hdf5 files. Length of this list
    is the length of the extracted dataset.

    Args:
        file_paths (list): List of file paths.
        file_shapes (list): List of file shapes.
        n_frames (int): Number of frames to load from each hdf5 file.
        frame_index_stride (int): Interval between frames to load.
        key (str, optional): Key of hdf5 dataset to grab data from. Defaults to "data/image".
        initial_frame_axis (int, optional): Axis to iterate over. Defaults to 0.
        additional_axes_iter (list, optional): Additional axes to iterate over in the dataset.
            Defaults to None.
        sort_files (bool, optional): Sort files by number. Defaults to True.
        overlapping_blocks (bool, optional): Will take n_frames from sequence, then move by 1.
            Defaults to False.
        limit_n_frames (int, optional): Limit the number of frames to load from each file. This
            means n_frames per data file will be used. These will be the first frames in the file.
            Defaults to None.

    Returns:
        list: List of tuples with indices to extract images from hdf5 files.
            (file_name, key, indices) with indices being a tuple of slices.

    Example:
        .. code-block:: python

            [
                (
                    "/folder/path_to_file.hdf5",
                    "data/image",
                    (range(0, 1), slice(None, 256, None), slice(None, 256, None)),
                ),
                (
                    "/folder/path_to_file.hdf5",
                    "data/image",
                    (range(1, 2), slice(None, 256, None), slice(None, 256, None)),
                ),
                ...,
            ]
    """
    if not limit_n_frames:
        limit_n_frames = np.inf

    assert len(file_paths) == len(file_shapes), "file_paths and file_shapes must have same length"

    if additional_axes_iter:
        # cannot contain initial_frame_axis
        assert initial_frame_axis not in additional_axes_iter, (
            "initial_frame_axis cannot be in additional_axes_iter. "
            "We are already iterating over that axis."
        )
    else:
        additional_axes_iter = []

    if sort_files:
        try:
            # this is like an np.argsort, returns the indices that would sort the array
            indices_sorting_file_paths = sorted(
                range(len(file_paths)),
                key=lambda i: int(re.findall(r"\d+", file_paths[i])[-2]),
            )
            file_paths = [file_paths[i] for i in indices_sorting_file_paths]
            file_shapes = [file_shapes[i] for i in indices_sorting_file_paths]
        except Exception:
            log.warning("H5Generator: Could not sort file_paths by number.")

    # block size with stride included
    block_size = n_frames * frame_index_stride

    if not overlapping_blocks:
        block_step_size = block_size
    else:
        # now blocks overlap by n_frames - 1
        block_step_size = 1

    def axis_indices_files():
        # For every file
        for shape in file_shapes:
            n_frames_in_file = shape[initial_frame_axis]
            # Optionally limit frames to load from each file
            n_frames_in_file = min(n_frames_in_file, limit_n_frames)
            indices = [
                list(range(i, i + block_size, frame_index_stride))
                for i in range(0, n_frames_in_file - block_size + 1, block_step_size)
            ]
            yield [indices]

    indices = []
    skipped_files = 0
    for file, shape, axis_indices in zip(file_paths, file_shapes, list(axis_indices_files())):
        # remove all the files that have empty list at initial_frame_axis
        # this can happen if the file is too small to fit a block
        if not axis_indices[0]:  # initial_frame_axis is the first entry in axis_indices
            skipped_files += 1
            continue

        if additional_axes_iter:
            axis_indices += [list(range(shape[axis])) for axis in additional_axes_iter]

        axis_indices = product(*axis_indices)

        for axis_index in axis_indices:
            full_indices = [slice(size) for size in shape]
            for i, axis in enumerate([initial_frame_axis] + list(additional_axes_iter)):
                full_indices[axis] = axis_index[i]
            indices.append((file, key, tuple(full_indices)))

    if skipped_files > 0:
        log.warning(
            f"H5Generator: Skipping {skipped_files} files with not enough frames "
            f"which is about {skipped_files / len(file_paths) * 100:.2f}% of the "
            f"dataset. This can be fine if you expect set `n_frames` and "
            "`frame_index_stride` to be high. Minimum frames in a file needs to be at "
            f"least n_frames * frame_index_stride = {n_frames * frame_index_stride}. "
        )

    return indices


def _h5_reopen_on_io_error(
    dataloader_obj: H5FileHandleCache,
    file,
    key,
    indices,
    retry_count,
    **kwargs,
):
    """Reopen the file if an I/O error occurs.
    Also removes the file from the cache and try to close file.
    """
    file_name = indices[0]
    try:
        file_handle = dataloader_obj._file_handle_cache.pop(file_name, None)
        if file_handle is not None:
            file_handle.close()
    except Exception:
        pass

    log.warning(
        f"H5Generator: I/O error occurred while reading file {file_name}. "
        f"Retry opening file. Retry count: {retry_count}."
    )


class H5Generator(Dataset):
    """H5Generator class for iterating over hdf5 files in an advanced way.
    Mostly used internally, you might want to use the Dataloader class instead.
    Loads one item at a time. Always outputs numpy arrays.
    """

    def __init__(
        self,
        file_paths: List[str],
        key: str = "data/image",
        n_frames: int = 1,
        shuffle: bool = True,
        return_filename: bool = False,
        limit_n_samples: int | None = None,
        limit_n_frames: int | None = None,
        seed: int | None = None,
        cache: bool = False,
        additional_axes_iter: tuple | None = None,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        initial_frame_axis: int = 0,
        insert_frame_axis: bool = True,
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        validate: bool = True,
        **kwargs,
    ):
        super().__init__(file_paths, key, validate=validate, **kwargs)

        self.n_frames = int(n_frames)
        self.frame_index_stride = int(frame_index_stride)
        self.frame_axis = int(frame_axis)
        self.insert_frame_axis = insert_frame_axis
        self.initial_frame_axis = int(initial_frame_axis)
        self.return_filename = return_filename
        self.shuffle = shuffle
        self.sort_files = sort_files
        self.overlapping_blocks = overlapping_blocks
        self.limit_n_samples = limit_n_samples
        self.limit_n_frames = limit_n_frames
        self.seed = seed
        self.additional_axes_iter = additional_axes_iter or []

        assert self.frame_index_stride > 0, (
            f"`frame_index_stride` must be greater than 0, got {self.frame_index_stride}"
        )
        assert self.n_frames > 0, f"`n_frames` must be greater than 0, got {self.n_frames}"

        # Extract some general information about the dataset
        image_shapes = np.array(self.file_shapes)
        image_shapes = np.delete(
            image_shapes, (self.initial_frame_axis, *self.additional_axes_iter), axis=1
        )
        n_dims = len(image_shapes[0])

        self.equal_file_shapes = np.all(image_shapes == image_shapes[0])
        if not self.equal_file_shapes:
            log.warning(
                "H5Generator: Not all files have the same shape. "
                "This can lead to issues when resizing images later...."
            )
            self.shape = np.array([None] * n_dims)
        else:
            self.shape = np.array(image_shapes[0])

        if insert_frame_axis:
            _frame_axis = map_negative_indices([frame_axis], len(self.shape) + 1)
            self.shape = np.insert(self.shape, _frame_axis, 1)
        if self.shape[frame_axis]:
            self.shape[frame_axis] = self.shape[frame_axis] * n_frames

        # Set random number generator
        self.rng = np.random.default_rng(self.seed)

        self.indices = generate_h5_indices(
            file_paths=self.file_paths,
            file_shapes=self.file_shapes,
            n_frames=self.n_frames,
            frame_index_stride=self.frame_index_stride,
            key=self.key,
            initial_frame_axis=self.initial_frame_axis,
            additional_axes_iter=self.additional_axes_iter,
            sort_files=self.sort_files,
            overlapping_blocks=self.overlapping_blocks,
            limit_n_frames=self.limit_n_frames,
        )

        if not self.shuffle:
            log.warning("H5Generator: Not shuffling data.")

        if limit_n_samples:
            log.warning(
                f"H5Generator: Limiting number of samples to {limit_n_samples} "
                f"out of {len(self.indices)}"
            )
            self.indices = self.indices[:limit_n_samples]

        self.shuffled_items = list(range(len(self.indices)))

        # Retry count for I/O errors
        self.retry_count = 0

        # Create a cache for the data
        self.cache = cache
        self._data_cache = {}

    def _get_single_item(self, idx):
        # Check if the item is already in the cache
        if self.cache and idx in self._data_cache:
            return self._data_cache[idx]

        # Get the data
        file_name, key, indices = self.indices[idx]
        file = self.get_file(file_name)
        image = self.load(file, key, indices)
        file_data = json_dumps(
            {
                "fullpath": file.filename,
                "filename": file.stem,
                "indices": indices,
            }
        )

        if self.cache:
            # Store the image and file data in the cache
            self._data_cache[idx] = [image, file_data]

        return image, file_data

    def __getitem__(self, index):
        image, file_data = self._get_single_item(self.shuffled_items[index])

        if self.return_filename:
            return image, file_data
        else:
            return image

    @retry_on_io_error(
        max_retries=MAX_RETRY_ATTEMPTS,
        initial_delay=INITIAL_RETRY_DELAY,
        retry_action=_h5_reopen_on_io_error,
    )
    def load(
        self,
        file: File,
        key: str,
        indices: Tuple[Union[list, slice, int], ...] | List[int] | int | None = None,
    ):
        """Extract data from hdf5 file.
        Args:
            file_name (str): name of the file to extract image from.
            key (str): key of the hdf5 dataset to grab data from.
            indices (tuple): indices to extract image from (tuple of slices)
        Returns:
            np.ndarray: image extracted from hdf5 file and indexed by indices.
        """
        try:
            images = file.load_data(key, indices)
        except (OSError, IOError):
            # Let the decorator handle I/O errors
            raise
        except Exception as exc:
            # For non-I/O errors, provide detailed context
            raise ValueError(
                f"Could not load image at index {indices} "
                f"and file {file.name} of shape {file[key].shape}"
            ) from exc

        # stack frames along frame_axis
        if self.insert_frame_axis:
            # move frames axis to self.frame_axis
            initial_frame_axis = self.initial_frame_axis
            if self.additional_axes_iter:
                # offset initial_frame_axis if we have additional axes that are before
                # the initial_frame_axis
                additional_axes_before = sum(
                    axis < self.initial_frame_axis for axis in self.additional_axes_iter
                )
                initial_frame_axis = initial_frame_axis - additional_axes_before

            images = np.moveaxis(images, initial_frame_axis, self.frame_axis)
        else:
            # append frames to existing axis
            images = np.concatenate(images, axis=self.frame_axis)

        return images

    def _shuffle(self):
        self.rng.shuffle(self.shuffled_items)
        log.info("H5Generator: Shuffled data.")

    def __len__(self):
        return len(self.indices)

    def iterator(self):
        """Generator that yields images from the hdf5 files."""
        if self.shuffle:
            self._shuffle()
        for idx in range(len(self)):
            yield self[idx]

    def __iter__(self):
        """
        Generator that yields images from the hdf5 files.
        """
        return self.iterator()

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} at 0x{id(self):x}: "
            f"{len(self)} batches, n_frames={self.n_frames}, key='{self.key}', "
            f"shuffle={self.shuffle}, file_paths={len(self.file_paths)}>"
        )

    def __str__(self):
        return (
            f"H5Generator with {len(self)} batches from {len(self.file_paths)} files "
            f"(key='{self.key}')"
        )

    def summary(self):
        """Return a string with dataset statistics and per-directory breakdown."""
        total_samples = len(self.indices)
        file_names = [idx[0] for idx in self.indices]
        # Try to infer directories from file_names
        directories = sorted({str(Path(f).parent) for f in file_names})
        samples_per_dir = count_samples_per_directory(file_names, directories)

        parts = [f"H5Generator with {total_samples} total samples:"]
        for dir_path, count in samples_per_dir.items():
            percentage = (count / total_samples) * 100 if total_samples else 0
            parts.append(f"  {dir_path}: {count} samples ({percentage:.1f}%)")
        print("\n".join(parts))


class H5DataSource:
    """Thread-safe random-access data source for HDF5 files.

    Implements ``grain.RandomAccessDataSource`` protocol (``__getitem__``
    and ``__len__``) so it can be plugged directly into a
    ``grain.MapDataset`` pipeline.

    Each worker thread gets its own ``H5FileHandleCache`` via
    ``threading.local()`` so ``h5py`` file handles are never shared across
    threads.

    Args:
        file_paths: Path(s) to HDF5 directory(ies) or file(s).
        key: HDF5 dataset key, e.g. ``"data/image"``.
        n_frames: Number of consecutive frames per sample.
        frame_index_stride: Stride between frames.
        frame_axis: Axis along which frames are stacked in the output.
        insert_frame_axis: Whether to insert a new axis for frames.
        initial_frame_axis: Source axis that stores frames in the file.
        additional_axes_iter: Extra axes to iterate over.
        sort_files: Sort files numerically.
        overlapping_blocks: Allow overlapping frame blocks.
        limit_n_samples: Cap the number of samples.
        limit_n_frames: Cap frames loaded per file.
        validate: Validate dataset against the zea format.
    """

    def __init__(
        self,
        file_paths: List[str] | str,
        key: str = "data/image",
        n_frames: int = 1,
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        insert_frame_axis: bool = True,
        initial_frame_axis: int = 0,
        additional_axes_iter: tuple | None = None,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        limit_n_samples: int | None = None,
        limit_n_frames: int | None = None,
        validate: bool = True,
        **kwargs,
    ):
        self.key = key
        self.n_frames = int(n_frames)
        self.frame_index_stride = int(frame_index_stride)
        self.frame_axis = int(frame_axis)
        self.insert_frame_axis = insert_frame_axis
        self.initial_frame_axis = int(initial_frame_axis)
        self.additional_axes_iter = list(additional_axes_iter or [])

        assert self.frame_index_stride > 0, (
            f"`frame_index_stride` must be > 0, got {self.frame_index_stride}"
        )
        assert self.n_frames > 0, f"`n_frames` must be > 0, got {self.n_frames}"

        # Discover files and shapes (reuses Dataset machinery)
        _dataset = Dataset(file_paths, key, validate=validate, **kwargs)
        self.file_paths = _dataset.file_paths
        self.file_shapes = _dataset.file_shapes
        _dataset.close()

        # Compute per-sample index table
        self.indices = generate_h5_indices(
            file_paths=self.file_paths,
            file_shapes=self.file_shapes,
            n_frames=self.n_frames,
            frame_index_stride=self.frame_index_stride,
            key=self.key,
            initial_frame_axis=self.initial_frame_axis,
            additional_axes_iter=self.additional_axes_iter,
            sort_files=sort_files,
            overlapping_blocks=overlapping_blocks,
            limit_n_frames=limit_n_frames,
        )

        if limit_n_samples:
            log.info(f"H5DataSource: Limiting to {limit_n_samples} / {len(self.indices)} samples.")
            self.indices = self.indices[:limit_n_samples]

        # Compute output shape (same logic as H5Generator)
        image_shapes = np.array(self.file_shapes)
        image_shapes = np.delete(
            image_shapes, (self.initial_frame_axis, *self.additional_axes_iter), axis=1
        )
        n_dims = len(image_shapes[0])
        equal = np.all(image_shapes == image_shapes[0])
        self.shape = np.array(image_shapes[0] if equal else [None] * n_dims)

        if insert_frame_axis:
            _fa = map_negative_indices([frame_axis], len(self.shape) + 1)
            self.shape = np.insert(self.shape, _fa, 1)
        if self.shape[frame_axis]:
            self.shape[frame_axis] = self.shape[frame_axis] * n_frames

        # Thread-local file handle caches (one per thread)
        self._local = threading.local()

    # -- grain.RandomAccessDataSource protocol ---------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> np.ndarray:
        """Return a single sample as a numpy array. Thread-safe."""
        file_name, key, indices = self.indices[index]
        cache = self._get_cache()
        file = cache.get_file(file_name)
        return self._load(file, key, indices)

    def __repr__(self) -> str:
        return (
            f"H5DataSource(n_samples={len(self)}, n_files={len(self.file_paths)}, key='{self.key}')"
        )

    # -- internals -------------------------------------------------------------

    def _get_cache(self) -> H5FileHandleCache:
        """Return the file-handle cache for the current thread."""
        if not hasattr(self._local, "cache"):
            self._local.cache = H5FileHandleCache()
        return self._local.cache

    def _load(self, file: File, key: str, indices):
        """Read from an open HDF5 file and handle frame-axis logic."""
        try:
            images = file.load_data(key, indices)
        except (OSError, IOError):
            # Invalidate cache entry and retry once
            cache = self._get_cache()
            fname = file.filename
            cache._file_handle_cache.pop(fname, None)
            try:
                file.close()
            except Exception:
                pass
            file = cache.get_file(fname)
            images = file.load_data(key, indices)

        if self.insert_frame_axis:
            initial = self.initial_frame_axis
            if self.additional_axes_iter:
                initial -= sum(ax < self.initial_frame_axis for ax in self.additional_axes_iter)
            images = np.moveaxis(images, initial, self.frame_axis)
        else:
            images = np.concatenate(images, axis=self.frame_axis)

        return images

    def close(self):
        """Close file handles for the current thread."""
        cache = getattr(self._local, "cache", None)
        if cache is not None:
            cache.close()


def _numpy_translate(array, range_from, range_to):
    """Map values from ``range_from`` to ``range_to`` (pure numpy)."""
    left_min, left_max = range_from
    right_min, right_max = range_to
    scaled = (array - left_min) / (left_max - left_min)
    return right_min + scaled * (right_max - right_min)


def _numpy_resize(image, image_size, resize_type="resize", rng=None):
    """Resize a single image (H, W, ...) using pure numpy / skimage.

    Args:
        image: ndarray of shape ``(..., H, W, C)`` or ``(H, W, C)``.
        image_size: ``(target_H, target_W)``.
        resize_type: ``"resize"``, ``"center_crop"``, ``"random_crop"`` or
            ``"crop_or_pad"``.
        rng: ``numpy.random.Generator`` instance (needed for ``random_crop``).

    Returns:
        Resized ndarray.
    """
    th, tw = image_size
    h, w = image.shape[-3], image.shape[-2]

    if resize_type == "resize":
        from skimage.transform import resize as skimage_resize

        # Preserve leading dims — skimage resize works on full shape
        target_shape = list(image.shape)
        target_shape[-3] = th
        target_shape[-2] = tw
        return skimage_resize(image, target_shape, preserve_range=True, anti_aliasing=True).astype(
            image.dtype
        )

    elif resize_type == "center_crop":
        start_h = max((h - th) // 2, 0)
        start_w = max((w - tw) // 2, 0)
        return image[..., start_h : start_h + th, start_w : start_w + tw, :]

    elif resize_type == "random_crop":
        if rng is None:
            rng = np.random.default_rng()
        start_h = rng.integers(0, max(h - th, 0) + 1)
        start_w = rng.integers(0, max(w - tw, 0) + 1)
        return image[..., start_h : start_h + th, start_w : start_w + tw, :]

    elif resize_type == "crop_or_pad":
        # Center-crop then zero-pad to target
        cropped_h = min(h, th)
        cropped_w = min(w, tw)
        start_h = max((h - th) // 2, 0)
        start_w = max((w - tw) // 2, 0)
        cropped = image[..., start_h : start_h + cropped_h, start_w : start_w + cropped_w, :]

        if cropped_h == th and cropped_w == tw:
            return cropped

        # Pad
        pad_h_before = (th - cropped_h) // 2
        pad_h_after = th - cropped_h - pad_h_before
        pad_w_before = (tw - cropped_w) // 2
        pad_w_after = tw - cropped_w - pad_w_before
        n_extra = len(image.shape) - 3  # leading dims
        pad_width = [(0, 0)] * n_extra + [
            (pad_h_before, pad_h_after),
            (pad_w_before, pad_w_after),
            (0, 0),
        ]
        return np.pad(cropped, pad_width, mode="constant", constant_values=0)

    else:
        raise ValueError(
            f"Unsupported resize_type: '{resize_type}'. "
            "Choose from 'resize', 'center_crop', 'random_crop', 'crop_or_pad'."
        )

def _make_transfer_fn(device: str | None = None):
    """Build a closure that transfers a numpy array to a backend device.

    All imports and device resolution happen **once** when this function is
    called.  The returned closure captures the resolved objects directly,
    so the per-batch hot path has zero import overhead.

    Args:
        device: Target device string, e.g. ``"gpu:0"``, ``"cuda:1"``,
            ``"cpu"``.  ``None`` uses the framework default.

    Returns:
        ``(np.ndarray) -> tensor`` callable.
    """
    backend_name = keras.backend.backend()

    if backend_name == "jax":
        import jax
        import jax.numpy as jnp

        if device is not None:
            from zea.backend.jax import str_to_jax_device

            jax_device = str_to_jax_device(device)
            return lambda array: jax.device_put(jnp.asarray(array), jax_device)
        return lambda array: jax.device_put(jnp.asarray(array))

    elif backend_name == "torch":
        import torch

        if device is not None:
            torch_device = torch.device(device.replace("gpu", "cuda"))
        else:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return lambda array: torch.as_tensor(array).to(torch_device, non_blocking=True)

    elif backend_name == "tensorflow":
        import tensorflow as tf

        if device is not None:
            tf_device = device.replace("cuda", "gpu")

            def _tf_transfer(array):
                with tf.device(tf_device):
                    return tf.constant(array)

            return _tf_transfer
        return lambda array: tf.constant(array)

    else:
        # numpy or unknown backend – return as-is
        return lambda array: array


class _PrefetchToDevice:
    """Iterator wrapper that transfers batches to a device one step ahead.

    While the caller processes batch *N* on the GPU, this wrapper already
    initiates the host-to-device copy for batch *N+1* in a background
    thread.  This overlaps data transfer with computation, hiding the
    CPU-to-GPU latency.

    Args:
        iterator: The source iterator yielding numpy arrays.
        device: Target device string passed to :func:`_make_transfer_fn`.
        prefetch_size: How many batches to keep in flight (default 2).
    """

    def __init__(self, iterator, device: str | None = None, prefetch_size: int = 2):
        self._iterator = iterator
        self._prefetch_size = prefetch_size
        self._buffer: deque = deque()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._exhausted = False
        # Resolve imports and device once — the closure is import-free
        self._transfer = _make_transfer_fn(device)
        # Pre-fill the buffer
        self._fill()

    def _fill(self):
        """Submit transfer jobs until the buffer reaches *prefetch_size*."""
        while len(self._buffer) < self._prefetch_size and not self._exhausted:
            try:
                batch = next(self._iterator)
            except StopIteration:
                self._exhausted = True
                break
            future = self._executor.submit(self._transfer, batch)
            self._buffer.append(future)

    def __iter__(self):
        return self

    def __next__(self):
        self._fill()
        if not self._buffer:
            self._executor.shutdown(wait=False)
            raise StopIteration
        future = self._buffer.popleft()
        return future.result()

class Dataloader:
    """High-performance HDF5 dataloader built on `Grain <https://github.com/google/grain>`_.

    Replaces the legacy TF ``make_dataloader`` pipeline with true
    multi-threaded I/O.  The read path is:

    .. code-block:: text

        grain threads (N) → h5py (thread-local handles) → numpy → user

    The entire pipeline runs in numpy — no framework dependency until
    you feed tensors to your model.

    Args:
        file_paths: Path(s) to HDF5 directory(ies) or file(s).
        key: HDF5 dataset key.
        batch_size: Batch size. ``None`` disables batching.
        n_frames: Consecutive frames per sample.
        shuffle: Shuffle data each epoch.
        seed: Random seed for shuffling.
        limit_n_samples: Cap number of samples.
        limit_n_frames: Cap frames per file.
        drop_remainder: Drop last incomplete batch.
        image_size: ``(H, W)`` target size.
        resize_type: ``"resize"``, ``"center_crop"``, ``"random_crop"``
            or ``"crop_or_pad"``.
        image_range: Original value range of images, e.g. ``(-60, 0)``.
        normalization_range: Target value range, e.g. ``(0, 1)``.
        clip_image_range: Clip values to ``image_range`` before normalizing.
        dataset_repetitions: Repeat dataset N times (``None`` = infinite).
        additional_axes_iter: Extra axes to iterate over.
        sort_files: Sort files numerically.
        overlapping_blocks: Allow overlapping frame blocks.
        augmentation: A callable ``(np.ndarray) -> np.ndarray`` applied
            per-sample *after* normalization.
        initial_frame_axis: Source frame axis in the file.
        insert_frame_axis: Insert new frame axis.
        frame_index_stride: Stride between frames.
        frame_axis: Axis for stacking frames.
        validate: Validate dataset against the zea format.
        shard_index: Shard index for distributed training.
        num_shards: Total number of shards.
        num_threads: Threads for parallel reads (0 = main thread only).
        prefetch_buffer_size: Grain prefetch buffer per process.

    Example:
        .. code-block:: python

            loader = Dataloader(
                file_paths="/data/camus",
                key="data/image_sc",
                batch_size=32,
                image_range=(-60, 0),
                normalization_range=(0, 1),
                image_size=(256, 256),
            )
            for batch in loader:
                ...  # batch.shape == (32, 256, 256, 1)
    """

    def __init__(
        self,
        file_paths: List[str] | str,
        key: str = "data/image",
        batch_size: int | None = 16,
        n_frames: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
        limit_n_samples: int | None = None,
        limit_n_frames: int | None = None,
        drop_remainder: bool = False,
        image_size: tuple | None = None,
        resize_type: str | None = None,
        image_range: tuple | None = None,
        normalization_range: tuple | None = None,
        clip_image_range: bool = False,
        dataset_repetitions: int | None = None,
        additional_axes_iter: tuple | None = None,
        sort_files: bool = True,
        overlapping_blocks: bool = False,
        augmentation: callable = None,
        initial_frame_axis: int = 0,
        insert_frame_axis: bool = True,
        frame_index_stride: int = 1,
        frame_axis: int = -1,
        validate: bool = True,
        shard_index: int | None = None,
        num_shards: int = 1,
        num_threads: int = 16,
        prefetch_buffer_size: int = 500,
        **kwargs,
    ):
        import grain

        # ── Validation ────────────────────────────────────────────────
        if normalization_range is not None:
            assert image_range is not None, (
                "If normalization_range is set, image_range must be set too."
            )
        if num_shards > 1:
            assert shard_index is not None, "shard_index must be specified"
            assert 0 <= shard_index < num_shards

        # ── Store config ──────────────────────────────────────────────
        self.batch_size = batch_size
        self.image_size = image_size
        self.resize_type = resize_type or ("resize" if image_size else None)
        self.image_range = image_range
        self.normalization_range = normalization_range
        self.clip_image_range = clip_image_range
        self.augmentation = augmentation
        self.num_threads = num_threads
        self.prefetch_buffer_size = prefetch_buffer_size
        self.shuffle = shuffle

        # Grain requires a concrete seed for shuffle — generate one if needed
        if seed is None and shuffle:
            seed = int(np.random.default_rng().integers(0, 2**31))
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # ── Data source ───────────────────────────────────────────────
        self.source = H5DataSource(
            file_paths=file_paths,
            key=key,
            n_frames=n_frames,
            frame_index_stride=frame_index_stride,
            frame_axis=frame_axis,
            insert_frame_axis=insert_frame_axis,
            initial_frame_axis=initial_frame_axis,
            additional_axes_iter=additional_axes_iter,
            sort_files=sort_files,
            overlapping_blocks=overlapping_blocks,
            limit_n_samples=limit_n_samples,
            limit_n_frames=limit_n_frames,
            validate=validate,
            **kwargs,
        )

        # ── Build Grain pipeline ──────────────────────────────────────
        ds = grain.MapDataset.source(self.source)

        # Shuffle (before sharding, so every shard sees different order)
        if shuffle:
            ds = ds.shuffle(seed=seed)

        # Shard
        if num_shards > 1:
            ds = ds[shard_index::num_shards]

        # Per-sample transforms (executed in parallel threads)
        ds = ds.map(self._transform_sample)

        # Repeat
        if dataset_repetitions is not None:
            ds = ds.repeat(num_epochs=dataset_repetitions)

        # Batch
        if batch_size is not None:
            ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)

        self._map_dataset = ds

    @property
    def dataset(self):
        """The underlying ``grain.MapDataset``."""
        return self._map_dataset

    def to_iter_dataset(self):
        """Convert to a ``grain.IterDataset`` with prefetching.

        This is called automatically when you iterate, but you can call
        it explicitly if you want to hold onto the ``IterDataset`` object.
        """
        import grain

        return self._map_dataset.to_iter_dataset(
            grain.ReadOptions(
                num_threads=self.num_threads,
                prefetch_buffer_size=self.prefetch_buffer_size,
            )
        )

    def __iter__(self):
        it = self.to_iter_dataset()
        device = os.environ.get("CUDA_VISIBLE_DEVICES")
        if device is not None:
            return _PrefetchToDevice(it, device=device)
        return it

    def __len__(self):
        """Number of batches (or samples if unbatched)."""
        return len(self._map_dataset)

    def __repr__(self):
        return (
            f"<Dataloader: {len(self.source)} samples, "
            f"batch_size={self.batch_size}, "
            f"key='{self.source.key}', "
            f"shape={tuple(self.source.shape)}, "
            f"threads={self.num_threads}>"
        )

    def _transform_sample(self, image: np.ndarray) -> np.ndarray:
        """Apply all per-sample transforms. Runs in grain worker threads."""
        # Ensure channel dim exists (at least 3-D)
        if image.ndim < 3:
            image = image[..., np.newaxis]

        # Clip to image range
        if self.clip_image_range and self.image_range is not None:
            image = np.clip(image, self.image_range[0], self.image_range[1])

        # Resize
        if self.image_size is not None:
            rng = self._rng if self.resize_type == "random_crop" else None
            image = _numpy_resize(image, self.image_size, self.resize_type, rng=rng)

        # Normalize
        if self.normalization_range is not None:
            image = _numpy_translate(image, self.image_range, self.normalization_range)

        # Augmentation
        if self.augmentation is not None:
            image = self.augmentation(image)

        return image

    def summary(self):
        """Print dataset statistics and per-directory breakdown."""
        src = self.source
        total_samples = len(src)
        file_names = [idx[0] for idx in src.indices]
        directories = sorted({str(Path(f).parent) for f in file_names})
        samples_per_dir = count_samples_per_directory(file_names, directories)

        parts = [f"Dataloader with {total_samples} total samples:"]
        for dir_path, count in samples_per_dir.items():
            pct = (count / total_samples) * 100 if total_samples else 0
            parts.append(f"  {dir_path}: {count} samples ({pct:.1f}%)")
        print("\n".join(parts))

    def close(self):
        """Release file handles."""
        self.source.close()
