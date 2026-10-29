import numpy as np


def predict_broadcast_shape(shape_a: tuple, shape_b: tuple) -> tuple:
    """
    Return the shape that results from broadcasting shapes `shape_a`
    and `shape_b` together, following the rule from Module 5:
    align from the trailing axis; two dimensions are compatible if they
    are equal or one is 1; a missing leading dimension counts as 1.

    Raise ValueError if the shapes are incompatible.
    Implement the rule yourself — do not call np.broadcast_shapes,
    np.broadcast_arrays, or perform an actual array operation.
    Examples:
        (3, 1) with (1, 4) -> (3, 4)
        (5,)   with (2, 5) -> (2, 5)
        (3,)   with (4,)   -> raises ValueError
    """
    x=max(len(shape_a),len(shape_b))
    shape_a=(1,)*(x-len(shape_a))+shape_a
    shape_b=(1,)*(x-len(shape_b))+shape_b
    result=[]
    for i,j in zip(shape_a,shape_b):
        if i==j:
            result.append(i)
        elif i==1 :
            result.append(j)
        elif j==1:
            result.append(i)
        else:
            raise ValueError("non comapatible")
    return tuple(result)
            
            
        


import numpy as np

def classify_by_memory(op, arr: np.ndarray) -> str:
    # 1. Apply the operation to the array
    result = op(arr)
    
    # 2. Check if they share the same memory block
    if np.shares_memory(arr, result):
        return "view"
    else:
        return "copy"
    
        
    
    


import numpy as np

def standardize_columns(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    `arr` is a 2D float array of shape (rows, cols). Return a new array
    of the same shape in which each COLUMN has been shifted to mean 0
    and scaled by its standard deviation:
        (arr - column_mean) / (column_std + eps)
    Use axis-based aggregation and broadcasting; no Python loops.
    A column with zero variance must give all zeros, not NaN.
    `arr` must not be modified.
    """
    # 1. Calculate 1D arrays for mean and std across rows (axis=0)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    
    # 2. Broadcast the 1D arrays across the 2D array.
    # The math operations automatically create and return a brand new array.
    # If a column has zero variance, (arr - mean) is 0. 0 / eps is exactly 0.
    return (arr - mean) / (std + eps)
