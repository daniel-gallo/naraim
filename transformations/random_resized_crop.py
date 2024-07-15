import tensorflow as tf

from transformations.transformation import Transformation


# Port of PyTorch's RandomResizedCrop
# https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomResizedCrop
#  * This has to be the ugliest code I have ever written, it is full of tf.cast :(
#  * Because this runs in graph mode, we cannot "early-exit" in the loop as in the PyTorch implementation
class RandomResizedCrop(Transformation):
    def __init__(self, size, scale, ratio):
        self.size = tf.constant([size, size])
        self.scale = tf.constant(scale)
        self.ratio = tf.constant(ratio)
        self.log_ratio = tf.math.log(ratio)

    def get_params(self, image):
        original_height = tf.shape(image)[0]
        original_width = tf.shape(image)[1]
        area = original_height * original_width

        # Fallback to central crop
        in_ratio = float(original_width) / float(original_height)
        if in_ratio < tf.math.reduce_min(self.ratio):
            new_width = original_width
            new_height = tf.cast(
                tf.math.round(
                    tf.cast(new_width, tf.float32) / tf.math.reduce_min(self.ratio)
                ),
                tf.int32,
            )
        elif in_ratio > tf.math.reduce_max(self.ratio):
            new_height = original_height
            new_width = tf.cast(
                tf.math.round(
                    tf.cast(new_height, tf.float32) * tf.math.reduce_max(self.ratio)
                ),
                tf.int32,
            )
        else:  # whole image
            new_width = original_width
            new_height = original_height
        i = (original_height - new_height) // 2
        j = (original_width - new_width) // 2

        # Try to perform a random resized crop
        for _ in range(10):
            target_area = tf.cast(area, tf.float32) * tf.random.uniform(
                shape=(),
                minval=self.scale[0],
                maxval=self.scale[1],
            )
            aspect_ratio = tf.math.exp(
                tf.random.uniform(
                    shape=(), minval=self.log_ratio[0], maxval=self.log_ratio[1]
                )
            )

            _new_width = tf.cast(
                tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32
            )
            _new_height = tf.cast(
                tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32
            )

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
        resized_crop = tf.image.resize(
            crop, self.size, method=tf.image.ResizeMethod.BICUBIC, antialias=True
        )
        return resized_crop
