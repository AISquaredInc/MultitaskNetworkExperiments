from sklearn.metrics import classification_report, confusion_matrix
import beyondml.tflow as mann
import tensorflow as tf
import numpy as np
import pickle
import os

if __name__ == '__main__':

    HIDDEN_LAYERS = 6
    HIDDEN_NODES = 1000
    
    (digit_x_train, digit_y_train), (digit_x_test, digit_y_test) = tf.keras.datasets.mnist.load_data()
    digit_x_train = digit_x_train.reshape((digit_x_train.shape[0], -1))/255
    digit_x_test = digit_x_test.reshape((digit_x_test.shape[0], -1))/255
    
    (fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()
    fashion_x_train = fashion_x_train.reshape((fashion_x_train.shape[0], -1))/255
    fashion_x_test = fashion_x_test.reshape((fashion_x_test.shape[0], -1))/255

    log_dir = os.path.join('.', 'logs', 'BothMNISTfc')
    digit_log_dir = os.path.join(log_dir, 'digitControl')
    fashion_log_dir = os.path.join(log_dir, 'fashionControl')
    mann_log_dir = os.path.join(log_dir, 'mann')

    callback = tf.keras.callbacks.EarlyStopping(
        min_delta = 0.01,
        patience = 3,
        restore_best_weights = True
    )

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

    input_layer = tf.keras.layers.Input(digit_x_train.shape[-1])
    x = tf.keras.layers.Dense(HIDDEN_NODES, activation = 'relu')(input_layer)
    for _ in range(HIDDEN_LAYERS - 1):
        x = tf.keras.layers.Dense(HIDDEN_NODES, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(10, activation = 'softmax')(x)
    model = tf.keras.models.Model(input_layer, output_layer)
    model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
    model.fit(digit_x_train, digit_y_train, epochs = 100, batch_size = 512, validation_split = 0.2, callbacks = [callback, digit_tboard], verbose = 0)
    digit_preds = model.predict(digit_x_test).argmax(axis = 1)
    print('Dedicated Model Digit Performance:')
    print(confusion_matrix(digit_y_test, digit_preds))
    print(classification_report(digit_y_test, digit_preds))
    model.save('digit_model.h5')

    input_layer = tf.keras.layers.Input(fashion_x_train.shape[-1])
    x = tf.keras.layers.Dense(HIDDEN_NODES, activation = 'relu')(input_layer)
    for _ in range(HIDDEN_LAYERS - 1):
        x = tf.keras.layers.Dense(HIDDEN_NODES, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(10, activation = 'softmax')(x)
    model = tf.keras.models.Model(input_layer, output_layer)
    model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
    model.fit(fashion_x_train, fashion_y_train, epochs = 100, batch_size = 512, validation_split = 0.2, callbacks = [callback, fashion_tboard], verbose = 0)
    fashion_preds = model.predict(fashion_x_test).argmax(axis = 1)
    print('Dedicated Model Fashion Performance:')
    print(confusion_matrix(fashion_y_test, fashion_preds))
    print(classification_report(fashion_y_test, fashion_preds))
    model.save('fashion_model.h5')

    digit_input = tf.keras.layers.Input(digit_x_train.shape[-1])
    fashion_input = tf.keras.layers.Input(fashion_x_train.shape[-1])

    x = mann.layers.MultiMaskedDense(HIDDEN_NODES, activation = 'relu')([digit_input, fashion_input])
    for _ in range(HIDDEN_LAYERS - 1):
        x = mann.layers.MultiMaskedDense(HIDDEN_NODES, activation = 'relu')(x)
    output_layer = mann.layers.MultiMaskedDense(10, activation = 'softmax')(x)

    model = tf.keras.models.Model([digit_input, fashion_input], output_layer)
    model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
    model = mann.utils.mask_model(model, 90, x = [digit_x_train, fashion_x_train], y = [digit_y_train.reshape(-1, 1), fashion_y_train.reshape(-1, 1)])
    model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

    model.fit(
        [digit_x_train, fashion_x_train],
        [digit_y_train, fashion_y_train],
        epochs = 100,
        batch_size = 512,
        validation_split = 0.2,
        callbacks = [callback, mann_tboard],
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

    new_digit_input = tf.keras.layers.Input(digit_x_train.shape[1:])
    new_fashion_input = tf.keras.layers.Input(fashion_x_train.shape[1:])
    x = mann.layers.SparseMultiDense.from_layer(model.layers[2])([new_digit_input, new_fashion_input])
    for i in range(HIDDEN_LAYERS - 1):
        x = mann.layers.SparseMultiDense.from_layer(model.layers[i + 3])(x)
    output_layer = mann.layers.SparseMultiDense.from_layer(model.layers[-1])(x)
    model = tf.keras.models.Model([new_digit_input, new_fashion_input], output_layer)
    
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
    
