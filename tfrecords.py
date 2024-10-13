"""
Create TFRecords from ImageNet dataset

Adapted from: https://keras.io/examples/keras_recipes/creating_tfrecords/
"""

import os
import random

import tensorflow as tf
from tqdm import tqdm


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, path, example):
    feature = {
        "image": image_feature(image),
        "path": bytes_feature(example["path"]),
        "class_id": int64_feature(example["class_id"]),
        "image_id": int64_feature(example["image_id"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "class_id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example


def main():
    data_split = "train"
    assert data_split in ["train", "val"]

    tfrecords_dir = f"tfrecords_imagenet_shuffled_{data_split}"
    images_dir = f"/scratch-nvme/ml-datasets/imagenet/ILSVRC/Data/CLS-LOC/{data_split}/"

    classes = set()
    datapoints = []
    if data_split == "val":
        with open(f"/scratch-nvme/ml-datasets/imagenet/LOC_{data_split}_solution.csv", "r") as fopen:
            for idx, line in enumerate(fopen.readlines()[1:]):  # skip header
                imname = line.strip().split(",")[0]
                cls = line.strip().split(",")[1].split()[0]
                path = imname  # no class folder for validation set
                datapoints.append((path, cls))
                classes.add(cls)
    else:
        # LOC_train_solution.csv does not contain the class labels for every datapoint, so we need a different approach
        for cls in os.listdir(images_dir):
            for imname in os.listdir(os.path.join(images_dir, cls)):
                path = f"{cls}/{imname.split('.')[0]}"
                datapoints.append((path, cls))
                classes.add(cls)

    classes = sorted(list(classes))
    class_mapping = {class_name: i for i, class_name in enumerate(classes)}
    annotations = [
        {"image_id": i, "class_id": class_mapping[cls], "path": path} for i, (path, cls) in enumerate(datapoints)
    ]

    # Shuffle datapoints
    random.shuffle(annotations)

    num_samples = 4096
    num_tfrecords = len(annotations) // num_samples

    if len(annotations) % num_samples:
        num_tfrecords += 1  # add one record if there are any remaining samples

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)  # creating TFRecords output folder

    print(
        f"\nWriting {len(annotations)} images to TFRecords, with {num_samples} images per .tfrec file this will be "
        f"{num_tfrecords} total records files.\n"
    )
    for tfrec_num in tqdm(range(num_tfrecords)):
        samples = annotations[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]
        with tf.io.TFRecordWriter(tfrecords_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))) as writer:
            for sample in tqdm(samples):
                image_path = images_dir + sample["path"] + ".JPEG"
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                if image.shape[-1] == 4:
                    image = image[:, :, :3]
                example = create_example(image, image_path, sample)
                writer.write(example.SerializeToString())

    # for key in features.keys():
    #     if key != "image":
    #         print(f"{key}: {features[key]}")
    #
    # print(f"Image shape: {features['image'].shape}")
    # plt.figure(figsize=(7, 7))
    # plt.imshow(features["image"].numpy())
    # plt.savefig('test.png')
    # plt.show()


if __name__ == "__main__":
    main()
