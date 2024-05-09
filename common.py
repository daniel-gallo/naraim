import jax.numpy as jnp
import jaxlib


# The leaves of these nested dictionaries are always jaxlib.xla_extension.ArrayImpl
def are_nested_dicts_equal(dict1, dict2):
    # Check if both inputs are dictionaries
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return False

    # Check if the keys are the same
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    # Recursively check each value
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]

        if isinstance(value1, jaxlib.xla_extension.ArrayImpl) and isinstance(
            value2, jaxlib.xla_extension.ArrayImpl
        ):
            # If both values are jaxlib.xla_extension.ArrayImpl objects, compare them using array_equal
            if not jnp.array_equal(value1, value2):
                return False
        elif isinstance(value1, dict) and isinstance(value2, dict):
            # If both values are dictionaries, recursively check them
            if not are_nested_dicts_equal(value1, value2):
                return False
        else:
            # If values are neither arrays nor dictionaries, compare them directly
            if value1 != value2:
                return False

    # If all checks passed, the nested dictionaries are equal
    return True
