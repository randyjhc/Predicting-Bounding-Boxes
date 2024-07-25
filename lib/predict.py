import os, re, time, json, zipfile, wget


import PIL.Image, PIL.ImageFont, PIL.ImageDraw
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
from lib.final_model import *
import gdown
import wget

BATCH_SIZE = 64


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


def get_visualization_training_dataset(data_dir):
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


def get_visualization_validation_dataset(data_dir):
    dataset = tfds.load(
        "caltech_birds2010", split="test", data_dir=data_dir, download=False
    )
    visualization_validation_dataset = dataset.map(
        read_image_tfds_with_original_bbox, num_parallel_calls=16
    )
    return visualization_validation_dataset


def get_training_dataset(dataset):
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    dataset = dataset.shuffle(512, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(-1)
    return dataset


def get_validation_dataset(dataset):
    dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    return dataset


def train():
    print("Start Preprocessing...")
    # Download the dataset
    # !wget https://storage.googleapis.com/tensorflow-3-public/datasets/caltech_birds2010_011.zip
    # url = "https://storage.googleapis.com/tensorflow-3-public/datasets/caltech_birds2010_011.zip"
    # wget.download(url)

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

    visualization_training_dataset = get_visualization_training_dataset(data_dir)
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

    visualization_validation_dataset = get_visualization_validation_dataset(data_dir)
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
    2.3 Load and prepare the datasets for the model
    """
    # BATCH_SIZE = 64
    training_dataset = get_training_dataset(visualization_training_dataset)
    validation_dataset = get_validation_dataset(visualization_validation_dataset)

    """
    3. Define the Network
    """
    # define your model
    model = define_and_compile_model()
    # print model layers
    model.summary()

    """
    4. Train the Model
    """
    # You'll train 50 epochs
    EPOCHS = 2

    ### START CODE HERE ###

    # Choose a batch size
    BATCH_SIZE = 64

    # Get the length of the training set
    length_of_training_dataset = len(visualization_training_dataset)
    print(length_of_training_dataset)

    # Get the length of the validation set
    length_of_validation_dataset = len(visualization_validation_dataset)
    print(length_of_validation_dataset)

    # Get the steps per epoch (may be a few lines of code)
    steps_per_epoch = length_of_training_dataset // BATCH_SIZE
    if length_of_training_dataset % BATCH_SIZE > 0:
        steps_per_epoch += 1

    # get the validation steps (per epoch) (may be a few lines of code)
    validation_steps = length_of_validation_dataset // BATCH_SIZE
    if length_of_validation_dataset % BATCH_SIZE > 0:
        validation_steps += 1

    ### END CODE HERE

    ### YOUR CODE HERE ####

    # Define the Keras TensorBoard callback.
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # Fit the model, setting the parameters noted in the instructions above.
    history = model.fit(
        x=training_dataset,
        # y=None,
        # batch_size=None,
        epochs=EPOCHS,
        # epochs=1,
        # verbose='auto',
        callbacks=[tensorboard_callback],
        # validation_split=0.0,
        validation_data=validation_dataset,
        # shuffle=True,
        # class_weight=None,
        # sample_weight=None,
        # initial_epoch=0,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        # validation_batch_size=None,
        # validation_freq=1
    )

    ### END CODE HERE ###


def intersection_over_union(pred_box, true_box):

    xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(pred_box, 4, axis=1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis=1)

    # Calculate coordinates of overlap area between boxes
    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    # Calculates area of true and predicted boxes
    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    # Calculates overlap area and union area.
    overlap_area = np.maximum((xmax_overlap - xmin_overlap), 0) * np.maximum(
        (ymax_overlap - ymin_overlap), 0
    )
    union_area = (pred_box_area + true_box_area) - overlap_area

    # Defines a smoothing factor to prevent division by 0
    smoothing_factor = 1e-10

    # Updates iou score
    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)

    return iou

def preview():
    print("Start Previewing...")
    # Specify the data directory
    data_dir = "./data"
    # Create the data directory
    try:
        os.mkdir(data_dir)
    except FileExistsError:
        print(f"{data_dir} already exists")

    # Load the dataset
    ds_name = 'caltech_birds2010_011.zip'
    if not os.path.isfile(ds_name):
        url = "https://storage.googleapis.com/tensorflow-3-public/datasets/caltech_birds2010_011.zip"
        wget.download(url)
        # Extract the dataset into the data directory
        with zipfile.ZipFile("./caltech_birds2010_011.zip") as zipref:
            zipref.extractall(data_dir)
    
    # Random choose 10 images for preview
    visualization_validation_dataset = get_visualization_validation_dataset(data_dir)
    original_images, normalized_images, normalized_bboxes = (
        dataset_to_numpy_with_original_bboxes_util(
            visualization_validation_dataset, N=500
        )
    )
    indexes = np.random.choice(len(original_images), size=10)
    # print(list(zip(original_images[indexes], normalized_bboxes[indexes])))
    # return original_images[indexes]
    images = [(original_images[i], " ".join(map(str, normalized_bboxes[i]))) for i in indexes]
    return images

def predict(selected_images):
    print(f"Start Predicting...")
    
    data_dir = "./data"

    # Load the model
    model_name = 'class_model.h5'
    if not os.path.isfile(model_name):
        file_id = '1kUjC7aiBz1CWtohpL7xbF3igA72t8Dez'
        # url = f'https://docs.google.com/uc?id={file_id}'
        url = f'https://docs.google.com/uc?export=download&id={file_id}'
        print(f'Downloading {url}')
        wget.download(url, out=model_name)
        
    model = tf.keras.models.load_model(model_name, compile=False)

    visualization_validation_dataset = get_visualization_validation_dataset(data_dir)

    # Makes predictions
    # original_images, normalized_images, normalized_bboxes = dataset_to_numpy_with_original_bboxes_util(visualization_validation_dataset, N=500)
    original_images, normalized_images, normalized_bboxes = (
        dataset_to_numpy_with_original_bboxes_util(
            visualization_validation_dataset, N=500
        )
    )
    predicted_bboxes = model.predict(normalized_images.astype("float32"))

    # Calculates IOU and reports true positives and false positives based on IOU threshold
    iou = intersection_over_union(predicted_bboxes, normalized_bboxes)
    iou_threshold = 0.5

    print(
        "Number of predictions where iou > threshold(%s): %s"
        % (iou_threshold, (iou >= iou_threshold).sum())
    )
    print(
        "Number of predictions where iou < threshold(%s): %s"
        % (iou_threshold, (iou < iou_threshold).sum())
    )
    
    # plot_metrics("loss", "Bounding Box Loss", ylim=0.2)

    """
    7. Visualize Predications
    """
    n = 10
    indexes = np.random.choice(len(predicted_bboxes), size=n)

    iou_to_draw = iou[indexes]
    norm_to_draw = original_images[indexes]
    
    return display_digits_with_boxes(
        original_images[indexes],
        predicted_bboxes[indexes],
        normalized_bboxes[indexes],
        iou[indexes],
        "True and Predicted values",
        bboxes_normalized=True,
    )
    

def predict_v2(selected_images):
    print(f"Start Predicting...")
    # print(f"[predict_v2] type(selected_images) = {type(selected_images)}")
    # print(f"[predict_v2] selected_images = {selected_images}")
    if not selected_images:
        return
    # Load the model
    model_name = 'class_model.h5'
    if not os.path.isfile(model_name):
        file_id = '1kUjC7aiBz1CWtohpL7xbF3igA72t8Dez'
        # url = f'https://docs.google.com/uc?id={file_id}'
        url = f'https://docs.google.com/uc?export=download&id={file_id}'
        print(f'Downloading {url}')
        wget.download(url, out=model_name)
        
    model = tf.keras.models.load_model(model_name, compile=False)
    
    ds_original_images, ds_normalized_images, ds_normalized_bboxes = [], [], []
    for org_img, org_bbox in selected_images:
        org_img, nor_img, _ = read_image_with_shape(org_img, np.zeros(4,))
        ds_original_images.append(org_img)
        ds_normalized_images.append(nor_img)
        if not org_bbox:
            ds_normalized_bboxes.append([0., 0., 0., 0.])
        else:
            ds_normalized_bboxes.append(list(map(float, org_bbox.split())))
    if len(selected_images) < 2:
        original_images = np.array(ds_original_images)
    else:
        original_images = np.array(ds_original_images, dtype="object")
    normalized_images = np.array(ds_normalized_images, dtype="object")
    normalized_bboxes = np.array(ds_normalized_bboxes, dtype="object")
    print(f'[predict_v2] ds_original_images = {type(ds_original_images)}')
    print(f'[predict_v2] original_images = {type(original_images)}')

    predicted_bboxes = model.predict(normalized_images.astype("float32"))

    iou = intersection_over_union(predicted_bboxes, normalized_bboxes)

    print(f'[predict_v2] shape of original_images = {original_images.shape}')
    print(f'[predict_v2] shape of normalized_images = {normalized_images.shape}')
    print(f'[predict_v2] shape of normalized_bboxes = {normalized_bboxes.shape}')
    print(f'[predict_v2] shape of iou = {iou.shape}')

    return display_digits_with_boxes(
        original_images,
        predicted_bboxes,
        normalized_bboxes,
        iou,
        "True and Predicted values",
        bboxes_normalized=True,
    )