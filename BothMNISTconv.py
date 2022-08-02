from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import beyondml.tflow as mann
import os
import pickle

if __name__ == '__main__':

    (digit_x_train, digit_y_train), (digit_x_test, digit_y_test) = tf.keras.datasets.mnist.load_data()
    digit_x_train = digit_x_train.reshape((digit_x_train.shape + (1,)))/255
    digit_x_test = digit_x_test.reshape((digit_x_test.shape + (1,)))/255
    
    (fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()
    fashion_x_train = fashion_x_train.reshape((fashion_x_train.shape + (1,)))/255
    fashion_x_test = fashion_x_test.reshape((fashion_x_test.shape + (1,)))/255

    log_dir = os.path.join('.', 'logs', 'BothMNISTconv')
    digit_log_dir = os.path.join(log_dir, 'digitControl')
    fashion_log_dir = os.path.join(log_dir, 'fashionControl')
    mann_log_dir = os.path.join(log_dir, 'mann')

    digit_tboard = tf.keras.callbacks.TensorBoard(
        log_dir = digit_log_dir,
        histogram_freq = 1
    )

    fashion_tboard = tf.keras.callbacks.TensorBoard(
        log_dir = fashion_log_dir,
        histogram_freq = 1
    )

    mann_tboard = tf.keras.callbacks.TensorBoard(
        log_dir = mann_log_dir,
        histogram_freq = 1
    )

    callback = tf.keras.callbacks.EarlyStopping(
        min_delta = 0.01,
        patience = 3,
        restore_best_weights = True
    )
    
    input_layer = tf.keras.layers.Input(digit_x_train.shape[1:])
    x = tf.keras.layers.Conv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )(input_layer)
    x = tf.keras.layers.Conv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(x)
    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(10, activation = 'softmax')(x)
    model = tf.keras.models.Model(
        input_layer,
        output_layer
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    model.fit(
        digit_x_train,
        digit_y_train,
        batch_size = 512,
        epochs = 100,
        validation_split = 0.2,
        callbacks = [callback, digit_tboard],
        verbose = 0
    )
    digit_preds = model.predict(digit_x_test).argmax(axis = 1)
    model.save('digit_model.h5')
    
    print('Dedicated Model Digit Performance:')
    print(confusion_matrix(digit_y_test, digit_preds))
    print(classification_report(digit_y_test, digit_preds))

    input_layer = tf.keras.layers.Input(fashion_x_train.shape[1:])
    x = tf.keras.layers.Conv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )(input_layer)
    x = tf.keras.layers.Conv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(x)
    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(10, activation = 'softmax')(x)
    model = tf.keras.models.Model(
        input_layer,
        output_layer
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    model.fit(
        fashion_x_train,
        fashion_y_train,
        batch_size = 512,
        epochs = 100,
        validation_split = 0.2,
        callbacks = [callback, fashion_tboard],
        verbose = 0
    )
    fashion_preds = model.predict(fashion_x_test).argmax(axis = 1)
    model.save('fashion_model.h5')

    print('Dedicated Model Fashion Performance:')
    print(confusion_matrix(fashion_y_test, fashion_preds))
    print(classification_report(fashion_y_test, fashion_preds))

    digit_input = tf.keras.layers.Input(digit_x_train.shape[1:])
    fashion_input = tf.keras.layers.Input(fashion_x_train.shape[1:])
    x = mann.layers.MultiMaskedConv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )([digit_input, fashion_input])
    x = mann.layers.MultiMaskedConv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    digit_selector1 = mann.layers.SelectorLayer(0)(x)
    fashion_selector1 = mann.layers.SelectorLayer(1)(x)
    digit_maxpool1 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(digit_selector1)
    fashion_maxpool1 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(fashion_selector1)
    x = mann.layers.MultiMaskedConv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )([digit_maxpool1, fashion_maxpool1])
    x = mann.layers.MultiMaskedConv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    digit_selector2 = mann.layers.SelectorLayer(0)(x)
    fashion_selector2 = mann.layers.SelectorLayer(1)(x)
    digit_maxpool2 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(digit_selector2)
    fashion_maxpool2 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(fashion_selector2)
    digit_flatten = tf.keras.layers.Flatten()(digit_maxpool2)
    fashion_flatten = tf.keras.layers.Flatten()(fashion_maxpool2)
    x = mann.layers.MultiMaskedDense(256, activation = 'relu')([digit_flatten, fashion_flatten])
    x = mann.layers.MultiMaskedDense(256, activation = 'relu')(x)
    output_layer = mann.layers.MultiMaskedDense(10, activation = 'softmax')(x)
    model = tf.keras.models.Model(
        [digit_input, fashion_input],
        output_layer
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam',
        loss_weights = [0.25, 0.75]
    )
    model = mann.utils.mask_model(
        model,
        80,
        method = 'gradients',
        x = [digit_x_train[:10000], fashion_x_train[:10000]],
        y = [digit_y_train[:10000].reshape(-1, 1), fashion_y_train[:10000].reshape(-1, 1)]
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )

    model.fit(
        [digit_x_train, fashion_x_train],
        [digit_y_train, fashion_y_train],
        epochs = 100,
        batch_size = 512,
        callbacks = [callback, mann_tboard],
        validation_split = 0.2,
        verbose = 0
    )

    model = mann.utils.utils.remove_layer_masks(model)
    model.save('combined_model.h5')
    preds = model.predict([digit_x_test, fashion_x_test])
    digit_preds = preds[0].argmax(axis = 1)
    fashion_preds = preds[1].argmax(axis = 1)
    
    print('Multitask Model Digit Performance:')
    print(confusion_matrix(digit_y_test, digit_preds))
    print(classification_report(digit_y_test, digit_preds))
    print('\n')
    print('Multitask Model Fashion Performance:')
    print(confusion_matrix(fashion_y_test, fashion_preds))
    print(classification_report(fashion_y_test, fashion_preds))

    digit_input = tf.keras.layers.Input(digit_x_train.shape[1:])
    fashion_input = tf.keras.layers.Input(fashion_x_train.shape[1:])
    x = mann.layers.SparseMultiConv.from_layer(model.layers[2])([digit_input, fashion_input])
    x = mann.layers.SparseMultiConv.from_layer(model.layers[3])(x)
    digit_selector1 = mann.layers.SelectorLayer(0)(x)
    fashion_selector1 = mann.layers.SelectorLayer(1)(x)
    digit_maxpool1 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(digit_selector1)
    fashion_maxpool1 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(fashion_selector1)
    x = mann.layers.SparseMultiConv.from_layer(model.layers[8])([digit_maxpool1, fashion_maxpool1])
    x = mann.layers.SparseMultiConv.from_layer(model.layers[9])(x)
    digit_selector2 = mann.layers.SelectorLayer(0)(x)
    fashion_selector2 = mann.layers.SelectorLayer(1)(x)
    digit_maxpool2 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(digit_selector2)
    fashion_maxpool2 = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(fashion_selector2)
    digit_flatten = tf.keras.layers.Flatten()(digit_maxpool2)
    fashion_flatten = tf.keras.layers.Flatten()(fashion_maxpool2)
    x = mann.layers.SparseMultiDense.from_layer(model.layers[16])([digit_flatten, fashion_flatten])
    x = mann.layers.SparseMultiDense.from_layer(model.layers[17])(x)
    output_layer = mann.layers.SparseMultiDense.from_layer(model.layers[-1])(x)
    model = tf.keras.models.Model(
        [digit_input, fashion_input],
        output_layer
    )

    preds = model.predict([digit_x_test, fashion_x_test])
    digit_preds = preds[0].argmax(axis = 1)
    fashion_preds = preds[1].argmax(axis = 1)
    
    print('Sparse Multitask Model Digit Performance:')
    print(confusion_matrix(digit_y_test, digit_preds))
    print(classification_report(digit_y_test, digit_preds))
    print('\n')
    print('Sparse Multitask Model Fashion Performance:')
    print(confusion_matrix(fashion_y_test, fashion_preds))
    print(classification_report(fashion_y_test, fashion_preds))

    with open('sparse_model.pkl', 'wb') as f:
        pickle.dump(model, f)