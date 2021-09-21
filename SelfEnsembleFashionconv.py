from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import mann

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape((x_train.shape + (1,)))/255
    x_test = x_test.reshape((x_test.shape + (1,)))/255

    callback = tf.keras.callbacks.EarlyStopping(
        min_delta = 0.01,
        patience = 3,
        restore_best_weights = True
    )

    input_layer = tf.keras.layers.Input(x_train.shape[1:])
    x = mann.layers.MultiMaskedConv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )([input_layer]*5)
    x = mann.layers.MultiMaskedConv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    sel1 = mann.layers.SelectorLayer(0)(x)
    sel2 = mann.layers.SelectorLayer(1)(x)
    sel3 = mann.layers.SelectorLayer(2)(x)
    sel4 = mann.layers.SelectorLayer(3)(x)
    sel5 = mann.layers.SelectorLayer(4)(x)
    pool1 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(sel1)
    pool2 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(sel2)
    pool3 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(sel3)
    pool4 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(sel4)
    pool5 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(sel5)
    x = mann.layers.MultiMaskedConv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )([pool1, pool2, pool3, pool4, pool5])
    x = mann.layers.MultiMaskedConv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    sel1 = mann.layers.SelectorLayer(0)(x)
    sel2 = mann.layers.SelectorLayer(1)(x)
    sel3 = mann.layers.SelectorLayer(2)(x)
    sel4 = mann.layers.SelectorLayer(3)(x)
    sel5 = mann.layers.SelectorLayer(4)(x)
    pool1 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(sel1)
    pool2 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(sel2)
    pool3 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(sel3)
    pool4 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(sel4)
    pool5 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(sel5)
    flat1 = tf.keras.layers.Flatten()(pool1)
    flat2 = tf.keras.layers.Flatten()(pool2)
    flat3 = tf.keras.layers.Flatten()(pool3)
    flat4 = tf.keras.layers.Flatten()(pool4)
    flat5 = tf.keras.layers.Flatten()(pool5)
    x = mann.layers.MultiMaskedDense(256, activation = 'relu')(
        [
            flat1,
            flat2,
            flat3,
            flat4,
            flat5
        ]
    )
    x = mann.layers.MultiMaskedDense(256, activation = 'relu')(x)
    output_layer = mann.layers.MultiMaskedDense(10, activation = 'softmax')(x)
    model = tf.keras.models.Model(
        input_layer,
        output_layer
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    model = mann.utils.mask_model(
        model,
        80,
        method = 'gradients',
        exclusive = True,
        x = x_train,
        y = [y_train]*5
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    model.fit(
        x_train,
        [y_train] * 5,
        epochs = 100,
        batch_size = 512,
        callbacks = [callback],
        validation_split = 0.2
    )
