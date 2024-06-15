import tensorflow as tf

from transformations.native_aspect_ratio_resize import NativeAspectRatioResize


def test_native_aspect_ratio_resize():
    transformation = NativeAspectRatioResize(224, 14)
    image = tf.random.uniform(shape=(1000, 3000, 3), minval=0, maxval=1)
    new_image = transformation(image)

    assert tf.math.reduce_prod(new_image.shape) <= 3 * 224**2
    assert new_image.shape[0] % 14 == 0
    assert new_image.shape[1] % 14 == 0
