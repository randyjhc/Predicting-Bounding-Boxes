import os, re, time, json, zipfile

# import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf

# from matplotlib import pyplot as plt
import tensorflow_datasets as tfds

# import cv2
print("Tensorflow version " + tf.__version__)

from datetime import datetime

# from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
# %load_ext tensorboard

from lib.draw import *


def read_image_tfds(image, bbox):
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)

    factor_x = tf.cast(shape[1], tf.float32)
    factor_y = tf.cast(shape[0], tf.float32)

    image = tf.image.resize(
        image,
        (
            224,
            224,
        ),
    )

    image = image / 127.5
    image -= 1

    bbox_list = [
        bbox[0] / factor_x,
        bbox[1] / factor_y,
        bbox[2] / factor_x,
        bbox[3] / factor_y,
    ]

    return image, bbox_list


def read_image_with_shape(image, bbox):
    original_image = image

    image, bbox_list = read_image_tfds(image, bbox)

    return original_image, image, bbox_list


def read_image_tfds_with_original_bbox(data):
    image = data["image"]
    bbox = data["bbox"]

    shape = tf.shape(image)
    factor_x = tf.cast(shape[1], tf.float32)
    factor_y = tf.cast(shape[0], tf.float32)

    bbox_list = [
        bbox[1] * factor_x,
        bbox[0] * factor_y,
        bbox[3] * factor_x,
        bbox[2] * factor_y,
    ]
    return image, bbox_list


def dataset_to_numpy_util(dataset, batch_size=0, N=0):

    # eager execution: loop through datasets normally
    take_dataset = dataset.shuffle(1024)

    if batch_size > 0:
        take_dataset = take_dataset.batch(batch_size)

    if N > 0:
        take_dataset = take_dataset.take(N)

    if tf.executing_eagerly():
        ds_images, ds_bboxes = [], []
        for images, bboxes in take_dataset:
            ds_images.append(images.numpy())
            ds_bboxes.append(bboxes.numpy())

    return (np.array(ds_images, dtype="object"), np.array(ds_bboxes, dtype="object"))


def dataset_to_numpy_with_original_bboxes_util(dataset, batch_size=0, N=0):

    normalized_dataset = dataset.map(read_image_with_shape)
    if batch_size > 0:
        normalized_dataset = normalized_dataset.batch(batch_size)

    if N > 0:
        normalized_dataset = normalized_dataset.take(N)

    if tf.executing_eagerly():
        ds_original_images, ds_images, ds_bboxes = [], [], []

    for original_images, images, bboxes in normalized_dataset:
        ds_images.append(images.numpy())
        ds_bboxes.append(bboxes.numpy())
        ds_original_images.append(original_images.numpy())

    return (
        np.array(ds_original_images, dtype="object"),
        np.array(ds_images, dtype="object"),
        np.array(ds_bboxes, dtype="object"),
    )


def get_visualization_training_dataset():
    dataset, info = tfds.load(
        "caltech_birds2010",
        split="train",
        with_info=True,
        data_dir=data_dir,
        download=False,
    )
    print(info)
    visualization_training_dataset = dataset.map(
        read_image_tfds_with_original_bbox, num_parallel_calls=16
    )
    return visualization_training_dataset


def get_visualization_validation_dataset():
    dataset = tfds.load(
        "caltech_birds2010", split="test", data_dir=data_dir, download=False
    )
    visualization_validation_dataset = dataset.map(
        read_image_tfds_with_original_bbox, num_parallel_calls=16
    )
    return visualization_validation_dataset


def predict():
    print("Start Predicting...")
    """
    # Download the dataset
    # !wget https://storage.googleapis.com/tensorflow-3-public/datasets/caltech_birds2010_011.zip

    # Specify the data directory
    data_dir = "./data"

    # Create the data directory
    try:
        os.mkdir(data_dir)
    except FileExistsError:
        print(f"{data_dir} already exists")

    # Extract the dataset into the data directory
    with zipfile.ZipFile("./caltech_birds2010_011.zip") as zipref:
        zipref.extractall(data_dir)
    
    config_plt()

    visualization_training_dataset = get_visualization_training_dataset()
    (visualization_training_images, visualization_training_bboxes) = (
        dataset_to_numpy_util(visualization_training_dataset, N=10)
    )
    display_digits_with_boxes(
        np.array(visualization_training_images),
        np.array([]),
        np.array(visualization_training_bboxes),
        np.array([]),
        "training images and their bboxes",
    )

    visualization_validation_dataset = get_visualization_validation_dataset()
    (visualization_validation_images, visualization_validation_bboxes) = (
        dataset_to_numpy_util(visualization_validation_dataset, N=10)
    )
    display_digits_with_boxes(
        np.array(visualization_validation_images),
        np.array([]),
        np.array(visualization_validation_bboxes),
        np.array([]),
        "validation images and their bboxes",
    )
    """
