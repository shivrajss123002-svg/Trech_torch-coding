
import numpy as np


def make_tensor_record(
    data: np.ndarray, device: str = "cpu", requires_grad: bool = False
) -> dict:
    """
    Create a tensor record.
    Data is stored as given (no copy).
    """
    return {
        "data": data,
        "device": device,
        "requires_grad": requires_grad
    }


def add_records(a: dict, b: dict) -> dict:
    """
    Add two tensor records.
    Raise ValueError if devices differ.
    """
    if a["device"] != b["device"]:
        raise ValueError("Different devices")

    new_rec = {
        "data": a["data"] + b["data"],
        "device": a["device"],
        "requires_grad": a["requires_grad"] or b["requires_grad"]
    }

    return new_rec


def detach_record(rec: dict) -> dict:
    """
    Return a new record with requires_grad=False.
    Share the same data buffer.
    """
    new_rec = rec.copy()
    new_rec["requires_grad"] = False

    return new_rec


def to_device_record(rec: dict, device: str) -> dict:
    """
    Return a record on the requested device.
    If device is same, return original record.
    Otherwise copy the data independently.
    """
    if rec["device"] == device:
        return rec

    new_rec = {
        "data": rec["data"].copy(),
        "device": device,
        "requires_grad": rec["requires_grad"]
    }

    return new_rec
