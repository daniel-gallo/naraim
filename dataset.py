import os
from glob import glob

import tensorflow as tf
from matplotlib import pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def resize_and_crop_image(image, patch_size):
    # extract the true height and width of the image (they are None when implicit)
    H, W, _ = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]
    # compute the sqrt of the aspect ratio
    sqrt_ratio = tf.cast(tf.sqrt(H / W), tf.float32)
    # compute the new height and width
    H = tf.cast(224 * sqrt_ratio, tf.int32)
    W = tf.cast(224**2 / tf.cast(H, tf.float32), tf.int32)
    # resize the image, now the num pixels is ~= 224^2
    image = tf.image.resize(image, [H, W])

    h, w = tf.shape(image)[0], tf.shape(image)[1]
    target_height = h - (h % patch_size)
    target_width = w - (w % patch_size)
    offset_height = (h - target_height) // 2
    offset_width = (w - target_width) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, target_height, target_width
    )
    return image


def patchify(image, patch_size):
    H, W, C = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]
    # calculate the number of patches for this specific image
    P_h = tf.cast(H / patch_size, tf.int32)
    P_w = tf.cast(W / patch_size, tf.int32)
    # crop away any pixels that don't fit into patches
    image = image[: P_h * patch_size, : P_w * patch_size]
    # [P_h, patch_size, P_w, patch_size, C]
    image = tf.reshape(image, [P_h, patch_size, P_w, patch_size, C])
    # [P_h, P_w, patch_size, patch_size, C]
    image = tf.transpose(image, [0, 2, 1, 3, 4])
    # [P_h*P_w, patch_size, patch_size, C]
    image = tf.reshape(image, [-1, patch_size, patch_size, C])
    # seq_len = P_h*P_w
    seq_len = tf.shape(image)[0]
    # [seq_len, patch_dim] where patch_dim = patch_size*patch_size*C
    image = tf.reshape(image, [seq_len, -1])

    image_coords = tf.stack(
        [
            tf.repeat(tf.range(P_h, dtype=tf.int32), (P_w,)),
            tf.tile(tf.range(P_w, dtype=tf.int32), (P_h,)),
        ],
        axis=-1,
    )
    return image, image_coords


def get_attention_matrix(prefix, max_seq_length):
    lower_triangular = tf.experimental.numpy.tril(
        tf.ones((max_seq_length, max_seq_length))
    )

    square = tf.pad(
        tensor=tf.ones((prefix, prefix)),
        paddings=[(0, max_seq_length - prefix), (0, max_seq_length - prefix)],
    )

    return tf.clip_by_value(lower_triangular + square, 0, 1)


def get_loss_mask(prefix, seq_length, max_seq_length):
    zeros_start = tf.zeros(prefix - 1)
    ones = tf.ones(seq_length - prefix)
    zeros_end = tf.zeros(max_seq_length - seq_length + 1)

    return tf.concat([zeros_start, ones, zeros_end], axis=0)


rng = tf.random.Generator.from_seed(123, alg="philox")


def augment_image(image, rng):
    seed = rng.make_seeds(1)[:, 0]

    # image = tf.image.stateless_random_contrast(image, lower=0.8, upper=1.2, seed=seed)
    # image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seed)
    # image = tf.image.stateless_random_saturation(image, lower=0.8, upper=1.2, seed=seed)
    image = tf.image.stateless_random_flip_left_right(image, seed)

    return image


def read_labeled_tfrecord(example, patch_size, rng):
    feature = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "class_id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example, feature)
    image = decode_image(example["image"])
    image = augment_image(image, rng)

    image = resize_and_crop_image(image, patch_size)
    patches, patch_indices = patchify(image, patch_size)
    seq_length = tf.shape(patches)[0]

    max_seq_len = (224 // patch_size) ** 2
    patches, _ = pad_sequence(patches, max_seq_len)
    patch_indices, _ = pad_sequence(patch_indices, max_seq_len)
    label = tf.cast(example["class_id"], tf.int32)

    prefix = tf.experimental.numpy.random.randint(
        low=1, high=seq_length, dtype=tf.experimental.numpy.int32
    )

    attention_matrix = get_attention_matrix(prefix, max_seq_len)
    loss_mask = get_loss_mask(prefix, seq_length, max_seq_len)

    return patches, patch_indices, label, attention_matrix, loss_mask


def pad_sequence(seq, seq_len):
    unpadded_seq_len = tf.shape(seq)[0]
    seq_dim = tf.shape(seq)[1]
    padding_mask_false = tf.zeros(unpadded_seq_len, dtype=tf.bool)
    padding_mask_true = tf.ones(seq_len - unpadded_seq_len, dtype=tf.bool)
    padding_mask = tf.concat([padding_mask_false, padding_mask_true], axis=0)

    padding_length = seq_len - unpadded_seq_len
    padding_seq = tf.zeros([padding_length, seq_dim], dtype=seq.dtype)
    seq = tf.concat([seq, padding_seq], axis=0)

    return seq, padding_mask


def load_dataset(filenames, patch_size):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)

    # Create a random number generator for data augmentations
    rng = tf.random.Generator.from_seed(42, alg="philox")
    dataset = dataset.map(
        lambda x: read_labeled_tfrecord(x, patch_size, rng), num_parallel_calls=AUTOTUNE
    )
    return dataset


if __name__ == "__main__":
    batch_size = 4
    image_dir = "./tfrecords"
    train_files = glob(os.path.join(image_dir, "*.tfrec"))
    train_dataset = load_dataset(train_files, 14)
    train_ds = (
        train_dataset.shuffle(10 * batch_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
        .as_numpy_iterator()
    )

    for batch in train_ds:
        image, image_coords, label, attention_matrix, loss_mask = batch
        print(image.shape)
        print(image_coords.shape)
        print(label.shape)
        print(attention_matrix.shape)
        print(loss_mask.shape)

        plt.imshow(loss_mask)
        plt.show()
