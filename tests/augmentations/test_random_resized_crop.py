import pytest
import tensorflow as tf

from augmentations.random_resized_crop import RandomResizedCrop


@pytest.mark.parametrize("original_size", [(1000, 2000), (300, 100)])
@pytest.mark.parametrize("size", [224, 448])
@pytest.mark.parametrize("scale", [(0.4, 1.0), (0.1, 0.2), (2.0, 3.0)])
@pytest.mark.parametrize("ratio", [(0.75, 1.33), (0.1, 0.2), (2.0, 3.0)])
def test_random_resized_crop(original_size, size, scale, ratio):
    original_image = tf.random.uniform(shape=(*original_size, 3), minval=0, maxval=1)
    random_resized_crop = RandomResizedCrop(size, scale, ratio)
    transformed_image = random_resized_crop(original_image)

    assert transformed_image.shape == (size, size, 3)
