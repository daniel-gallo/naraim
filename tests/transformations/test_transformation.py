import pytest
import tensorflow as tf

from transformations.auto_augment import AutoAugment
from transformations.native_aspect_ratio_resize import NativeAspectRatioResize
from transformations.random_horizontal_flip import RandomHorizontalFlip

transformations = [
    AutoAugment(),
    NativeAspectRatioResize(square_size=224, patch_size=14),
    RandomHorizontalFlip(),
    # BICUBIC interpolation can make the output be outside of [0, 1]
    # RandomResizedCrop(size=224, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
]


@pytest.mark.parametrize("transformation", transformations)
def test_output_dtype_and_range(transformation):
    # The output should be a float32 in the range [0, 1]
    input_image = tf.random.uniform(shape=(1000, 2000, 3), minval=0, maxval=1)
    assert tf.math.reduce_min(input_image) >= 0
    assert tf.math.reduce_max(input_image) <= 1
    assert input_image.dtype == tf.float32

    output_image = transformation(input_image)

    assert tf.math.reduce_min(output_image) >= 0
    assert tf.math.reduce_max(output_image) <= 1
    assert output_image.dtype == tf.float32
