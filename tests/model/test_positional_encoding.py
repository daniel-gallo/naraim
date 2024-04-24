import pytest

from model.positional import get_2d_positional_embedding


def test_positional_encoding_with_odd_embedding_dimension_fails():
    embedding_dimension = 127
    height = 12
    width = 8

    with pytest.raises(AssertionError):
        get_2d_positional_embedding(embedding_dimension, height, width)


def test_positional_encoding():
    embedding_dimension = 128
    height = 12
    width = 8

    encoding = get_2d_positional_embedding(embedding_dimension, height, width)
    assert encoding.shape == (height, width, embedding_dimension)
