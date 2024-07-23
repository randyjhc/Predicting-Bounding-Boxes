import tensorflow as tf


def feature_extractor(inputs):
    ### YOUR CODE HERE ###

    # Create a mobilenet version 2 model object
    mobilenet_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )

    # pass the inputs into this modle object to get a feature extractor for these inputs
    feature_extractor = mobilenet_model(inputs)

    ### END CODE HERE ###

    # return the feature_extractor
    return feature_extractor


def dense_layers(features):
    ### YOUR CODE HERE ###

    # global average pooling 2d layer
    x = tf.keras.layers.GlobalAveragePooling2D()(features)

    # flatten layer
    x = tf.keras.layers.Flatten()(x)

    # 1024 Dense layer, with relu
    x = tf.keras.layers.Dense(1024, activation="relu")(x)

    # 512 Dense layer, with relu
    x = tf.keras.layers.Dense(512, activation="relu")(x)

    ### END CODE HERE ###

    return x


def bounding_box_regression(x):
    ### YOUR CODE HERE ###

    # Dense layer named `bounding_box`
    bounding_box_regression_output = tf.keras.layers.Dense(4, name="bounding_box")(x)

    ### END CODE HERE ###

    return bounding_box_regression_output


def final_model(inputs):
    ### YOUR CODE HERE ###

    # features
    feature_cnn = feature_extractor(inputs)

    # dense layers
    last_dense_layer = dense_layers(feature_cnn)

    # bounding box
    bounding_box_output = bounding_box_regression(last_dense_layer)

    # define the TensorFlow Keras model using the inputs and outputs to your model
    model = tf.keras.Model(inputs=inputs, outputs=bounding_box_output)

    ### END CODE HERE ###

    return model


def define_and_compile_model():

    ### YOUR CODE HERE ###

    # define the input layer
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # create the model
    model = final_model(inputs)

    # compile your model
    model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9), loss="mse")

    ### END CODE HERE ###

    return model
