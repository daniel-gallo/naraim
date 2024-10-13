import tensorflow as tf

from transformations.transformation import Transformation


class RandomCrop(Transformation):
    def __init__(self, scale: tuple, ratio: tuple, min_num_pixels: int):
        """
        Args:
            scale (tuple): the lower and upper bounds of the percentage of area the crop will preserve
            ratio (tuple): the lower and upper bounds of the aspect ratios
            min_num_pixels (int): the minimum number of pixels in the crop

        Example:
            RandomCrop(scale=(0.4, 1.0), ratio=(0.75, 1.33))
        """
        self.scale = tf.constant(scale)
        self.ratio = tf.constant(ratio)
        self.log_ratio = tf.math.log(ratio)
        self.min_num_pixels = tf.constant(min_num_pixels)

    def get_params(self, image):
        original_height = tf.shape(image)[0]
        original_width = tf.shape(image)[1]
        area = original_height * original_width

        # Fallback to no crop
        i = 0
        j = 0
        new_height = original_height
        new_width = original_width

        # Try to perform a crop
        for _ in range(10):
            target_area = tf.cast(area, tf.float32) * tf.random.uniform(
                shape=(),
                minval=self.scale[0],
                maxval=self.scale[1],
            )
            aspect_ratio = tf.math.exp(tf.random.uniform(shape=(), minval=self.log_ratio[0], maxval=self.log_ratio[1]))

            _new_width = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
            _new_height = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)

            if _new_height * _new_width < self.min_num_pixels:
                continue

            if 0 < _new_width <= original_width and 0 < _new_height <= original_height:
                i = tf.random.uniform(
                    shape=(),
                    minval=0,
                    maxval=original_height - _new_height + 1,
                    dtype=tf.int32,
                )
                j = tf.random.uniform(
                    shape=(),
                    minval=0,
                    maxval=original_width - _new_width + 1,
                    dtype=tf.int32,
                )

                new_height = _new_height
                new_width = _new_width

        return i, j, new_height, new_width

    def __call__(self, image):
        i, j, new_height, new_width = self.get_params(image)
        crop = image[i : i + new_height, j : j + new_width]
        return crop
