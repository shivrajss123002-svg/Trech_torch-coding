
def broadcast_shapes(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...] | None:
    # Step 1: Pad the shorter shape on the left
    if len(shape_a) < len(shape_b):
        shape_a = (1,) * (len(shape_b) - len(shape_a)) + shape_a
    elif len(shape_b) < len(shape_a):
        shape_b = (1,) * (len(shape_a) - len(shape_b)) + shape_b

    # Step 2: Calculate the result shape
    result = []

    for dim_a, dim_b in zip(shape_a, shape_b):
        # Dimensions are compatible if equal or either is 1
        if dim_a == dim_b:
            result.append(dim_a)
        elif dim_a == 1:
            result.append(dim_b)
        elif dim_b == 1:
            result.append(dim_a)
        else:
            return None

    return tuple(result)
