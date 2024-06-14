import tensorflow as tf

from augmentations.augmentation import Augmentation


# Port of PyTorch's RandomResizedCrop
# https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomResizedCrop
class RandomResizedCrop(Augmentation):
    def __init__(self, size, scale, ratio):
        self.size = tf.constant([size, size])
        self.scale = tf.constant(scale)
        self.ratio = tf.constant(ratio)
        self.log_ratio = tf.math.log(ratio)

    def get_params(self, image):
        original_height, original_width, num_channels = image.shape
        assert num_channels == 3
        area = original_height * original_width

        for _ in range(10):
            target_area = area * tf.random.uniform(
                shape=(),
                minval=self.scale[0],
                maxval=self.scale[1],
            )
            aspect_ratio = tf.math.exp(
                tf.random.uniform(
                    shape=(), minval=self.log_ratio[0], maxval=self.log_ratio[1]
                )
            )

            new_width = tf.cast(
                tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32
            )
            new_height = tf.cast(
                tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32
            )

            if 0 < new_width <= original_width and 0 < new_height <= original_height:
                i = tf.random.uniform(
                    shape=(),
                    minval=0,
                    maxval=original_height - new_height + 1,
                    dtype=tf.int32,
                )
                j = tf.random.uniform(
                    shape=(),
                    minval=0,
                    maxval=original_width - new_width + 1,
                    dtype=tf.int32,
                )

                return i, j, new_height, new_width

        # Fallback to central crop
        in_ratio = float(original_width) / float(original_height)
        if in_ratio < tf.math.reduce_min(self.ratio):
            new_width = original_width
            new_height = tf.cast(
                tf.math.round(new_width / tf.math.reduce_min(self.ratio)), tf.int32
            )
        elif in_ratio > tf.math.reduce_max(self.ratio):
            new_height = original_height
            new_width = tf.cast(
                tf.math.round(new_height * tf.math.reduce_max(self.ratio)), tf.int32
            )
        else:  # whole image
            new_width = original_width
            new_height = original_height
        i = (original_height - new_height) // 2
        j = (original_width - new_width) // 2
        return i, j, new_height, new_width

    def __call__(self, image):
        i, j, new_height, new_width = self.get_params(image)
        crop = image[i : i + new_height, j : j + new_width]
        resized_crop = tf.image.resize(
            crop, self.size, method=tf.image.ResizeMethod.BICUBIC
        )
        return resized_crop
