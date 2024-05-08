import jax.numpy as jnp

from common import are_nested_dicts_equal


def test_are_nested_dicts_equal():
    nested_dict1 = {
        "a": {"x": jnp.array([1, 2]), "y": jnp.array([3, 4])},
        "b": {"z": jnp.array([5, 6])},
    }
    nested_dict2 = {
        "a": {"x": jnp.array([1, 2]), "y": jnp.array([3, 4])},
        "b": {"z": jnp.array([5, 6])},
    }
    nested_dict3 = {
        "a": {"x": jnp.array([1, 2]), "y": jnp.array([3, 4])},
        "b": {"z": jnp.array([5, 7])},
    }

    assert are_nested_dicts_equal(nested_dict1, nested_dict2) is True
    assert are_nested_dicts_equal(nested_dict1, nested_dict3) is False
