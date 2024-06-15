import tensorflow as tf

from transformations.transformation import Transformation


class RandomHorizontalFlip(Transformation):
    def __call__(self, image):
        return tf.image.random_flip_left_right(image)
