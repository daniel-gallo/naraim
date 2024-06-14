import tensorflow as tf
from timm.data.auto_augment import rand_augment_transform

from augmentations.augmentation import Augmentation


class AutoAugment(Augmentation):
    def __init__(self):
        self.transform = rand_augment_transform(
            config_str="rand-m9-mstd0.5-inc1", hparams={}
        )

    def __call__(self, image):
        image_as_pil = tf.keras.preprocessing.image.array_to_img(image)
        augmented_image = self.transform(image_as_pil)
        return tf.convert_to_tensor(
            tf.keras.preprocessing.image.img_to_array(augmented_image) / 255
        )
