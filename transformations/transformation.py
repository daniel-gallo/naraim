from abc import abstractmethod

import tensorflow as tf


class Transformation:
    @abstractmethod
    def __call__(self, image: tf.Tensor) -> tf.Tensor:
        """
        Both the input and output should be between [0, 1]
        """
        raise NotImplementedError()
