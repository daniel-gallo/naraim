import tensorflow as tf

from transformations.transformation import Transformation


class NativeAspectRatioResize(Transformation):
    def __init__(self, square_size, patch_size):
        """
        The image will be
         1. Rescaled so that the area is smaller or equal square_size^2
         2. Cropped so that the sides are multiples of patch_size
        """
        self.square_size = square_size
        self.patch_size = patch_size

    def __call__(self, image):
        # extract the true height and width of the image (they are None when implicit)
        height, width, _ = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]
        # compute the sqrt of the aspect ratio
        sqrt_ratio = tf.cast(tf.sqrt(height / width), tf.float32)
        # compute the new height and width
        height = tf.cast(224 * sqrt_ratio, tf.int32)
        width = tf.cast(224**2 / tf.cast(height, tf.float32), tf.int32)
        # resize the image, now the num pixels is ~= 224^2
        image = tf.image.resize(
            image, [height, width], tf.image.ResizeMethod.BICUBIC, antialias=True
        )

        target_height = height - (height % self.patch_size)
        target_width = width - (width % self.patch_size)
        offset_height = (height - target_height) // 2
        offset_width = (width - target_width) // 2
        image = tf.image.crop_to_bounding_box(
            image, offset_height, offset_width, target_height, target_width
        )
        return image
