import tensorflow as tf

from augmentations.random_horizontal_flip import RandomHorizontalFlip


def test_random_horizontal_flip():
    image = tf.random.uniform(shape=(224, 224, 3), minval=0, maxval=1)
    random_horizontal_flip = RandomHorizontalFlip()
    new_image = random_horizontal_flip(image)

    assert tf.reduce_all(new_image == tf.reverse(image, axis=[1])) or tf.reduce_all(
        new_image == image
    )
