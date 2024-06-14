import tensorflow as tf
from timm.data.auto_augment import rand_augment_transform

from transformations.transformation import Transformation


class AutoAugment(Transformation):
    def __init__(self):
        self.transform = rand_augment_transform(
            config_str="rand-m9-mstd0.5-inc1", hparams={}
        )

    def __call__(self, image):
        # TODO(daniel-gallo): for some reason this does not seem to work :///
        @tf.numpy_function(Tout=tf.float32)
        def _run_in_eager_mode(image):
            image_as_pil = tf.keras.preprocessing.image.array_to_img(image)
            augmented_image = self.transform(image_as_pil)
            return tf.keras.preprocessing.image.img_to_array(augmented_image) / 255

        return _run_in_eager_mode(image)
