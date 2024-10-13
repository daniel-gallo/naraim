import tensorflow as tf

from transformations.transformation import Transformation


class AIMInference(Transformation):
    def __init__(self, resize_size, crop_size):
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.offset = (resize_size - crop_size) // 2

    def __call__(self, image):
        height = tf.cast(tf.shape(image)[0], tf.float32)
        width = tf.cast(tf.shape(image)[1], tf.float32)

        if height < width:
            factor = self.resize_size / height
        else:
            factor = self.resize_size / width

        new_height = tf.cast(height * factor, tf.int32)
        new_width = tf.cast(width * factor, tf.int32)

        resized_image = tf.image.resize(
            image, [new_height, new_width], method=tf.image.ResizeMethod.BICUBIC, antialias=True
        )

        cropped_image = tf.image.crop_to_bounding_box(resized_image, self.offset, self.offset, 224, 224)
        return cropped_image
