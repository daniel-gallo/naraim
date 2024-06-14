import tensorflow as tf

from augmentations.augmentation import Augmentation


class RandomHorizontalFlip(Augmentation):
    def __call__(self, image):
        return tf.image.random_flip_left_right(image)
