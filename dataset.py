import collections
import itertools

import jax
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


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
    lower_triangular = tf.experimental.numpy.tril(tf.ones((max_seq_length, max_seq_length)))

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


def read_labeled_tfrecord(example, patch_size, transformations):
    feature = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "class_id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example, feature)
    image = tf.cast(tf.image.decode_jpeg(example["image"], channels=3), tf.float32) / 255.0

    for transformation in transformations:
        image = transformation(image)

    patches, patch_indices = patchify(image, patch_size)
    seq_length = tf.shape(patches)[0]

    max_seq_len = (224 // patch_size) ** 2
    patches, _ = pad_sequence(patches, max_seq_len)
    patch_indices, _ = pad_sequence(patch_indices, max_seq_len)
    label = tf.cast(example["class_id"], tf.int32)

    prefix = tf.experimental.numpy.random.randint(low=1, high=seq_length, dtype=tf.experimental.numpy.int32)

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


def load_dataset(filenames, patch_size, transformations):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)

    dataset = dataset.map(
        lambda x: read_labeled_tfrecord(x, patch_size, transformations),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset


def prefetch(iterator):
    # Prefetches the batches to the accelerator to avoid waiting in between iterations

    queue = collections.deque()

    def _prefetch(x):
        # Note that device_put is async
        return jax.device_put(x)

    def enqueue(n):
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree.map(_prefetch, data))

    enqueue(2)
    while queue:
        yield queue.popleft()
        enqueue(1)
