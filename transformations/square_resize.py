import tensorflow as tf

from transformations.transformation import Transformation


class SquareResize(Transformation):
    def __init__(self, size: int):
        self.size = tf.constant([size, size])

    def __call__(self, image):
        return tf.image.resize(
            image, self.size, method=tf.image.ResizeMethod.BICUBIC, antialias=True
        )
