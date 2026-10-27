import numpy as np


def basic_slice(a: np.ndarray, start: int, stop: int) -> np.ndarray:
    """a[start:stop] along the first axis."""
    return a[start:stop]


def boolean_mask(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """a[mask], mask a same-shape (or broadcastable) boolean array."""
    return a[mask]


def fancy_index(a: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """a[indices], indices an integer array."""
    return a[indices]
    


def is_a_view_of(child: np.ndarray, parent: np.ndarray) -> bool:
    """
    Whether `child`'s underlying memory overlaps `parent`'s, i.e.
    whether `child` is a VIEW of `parent` rather than an independent copy.
    """
    # TODO: np.shares_memory.
    return np.shares_memory(child,parent)
