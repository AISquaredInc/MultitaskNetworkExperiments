from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import beyondml.tflow as mann
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

    (boston_x_train, boston_y_train), (boston_x_test, boston_y_test) = tf.keras.datasets.boston_housing.load_data()
    boston_x_scaler = MinMaxScaler().fit(boston_x_train)
    boston_y_scaler = MinMaxScaler().fit(boston_y_train.reshape(-1, 1))

    boston_x_train = boston_x_scaler.transform(boston_x_train)
    boston_x_test = boston_x_scaler.transform(boston_x_test)
    boston_y_train = boston_y_scaler.transform(boston_y_train.reshape(-1, 1))
    boston_y_test = boston_y_scaler.transform(boston_y_test.reshape(-1, 1))
    
    callback = tf.keras.callbacks.EarlyStopping(
        min_delta = 0.01,
        patience = 3,
        restore_best_weights = True
    )

    log_dir = os.path.join('.', 'logs', 'BothMNISTBostonfc')
    boston_log_dir = os.path.join(log_dir, 'boston')
    mann_log_dir = os.path.join(log_dir, 'mann')

    boston_tboard = tf.keras.callbacks.TensorBoard(
        log_dir = boston_log_dir,
        histogram_freq = 1
    )

    mann_tboard = tf.keras.callbacks.TensorBoard(
        log_dir = mann_log_dir,
        histogram_freq = 1
    )

    # Boston housing control
    input_layer = tf.keras.layers.Input(boston_x_train.shape[1:])
    x = tf.keras.layers.Dense(digit_x_train.shape[-1], activation = 'relu')(input_layer)
    for _ in range(HIDDEN_LAYERS):
        x = tf.keras.layers.Dense(HIDDEN_NODES, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(1, activation = 'relu')(x)
    model = tf.keras.models.Model(input_layer, output_layer)
    model.compile(loss = 'mse', optimizer = 'adam')
    model.fit(boston_x_train, boston_y_train, epochs = 100, batch_size = 32, validation_split = 0.2, callbacks = [callback, boston_tboard], verbose = 0)
    boston_performance = model.evaluate(boston_x_test, boston_y_test, verbose = 0)
    print(f'Boston Dedicated Model Loss: {boston_performance}')
    model.save('boston_model.h5')

    # Multitask Model
    digit_input = tf.keras.layers.Input(digit_x_train.shape[1:])
    fashion_input = tf.keras.layers.Input(fashion_x_train.shape[1:])
    boston_input = tf.keras.layers.Input(boston_x_train.shape[1:])
    boston_reshape = mann.layers.MaskedDense(fashion_x_train.shape[-1], activation = 'relu')(boston_input)
    
    x = mann.layers.MultiMaskedDense(HIDDEN_NODES, activation = 'relu')([digit_input, fashion_input, boston_reshape])
    for _ in range(HIDDEN_LAYERS - 1):
        x = mann.layers.MultiMaskedDense(HIDDEN_NODES, activation = 'relu')(x)
        
    digit_selector = mann.layers.SelectorLayer(0)(x)
    fashion_selector = mann.layers.SelectorLayer(1)(x)
    boston_selector = mann.layers.SelectorLayer(2)(x)
    
    digit_output = mann.layers.MaskedDense(10, activation = 'softmax')(digit_selector)
    fashion_output = mann.layers.MaskedDense(10, activation = 'softmax')(fashion_selector)
    boston_output = mann.layers.MaskedDense(1, activation = 'relu')(boston_selector)

    model = tf.keras.models.Model([digit_input, fashion_input, boston_input], [digit_output, fashion_output, boston_output])
    model.compile(
        loss = [
            'sparse_categorical_crossentropy',
            'sparse_categorical_crossentropy',
            'mse'
        ],
        optimizer = 'adam'
    )
    model = mann.utils.mask_model(
        model,
        90,
        x = [
            digit_x_train[:boston_x_train.shape[0], :],
            fashion_x_train[:boston_x_train.shape[0], :],
            boston_x_train
        ],
        y = [
            digit_y_train[:boston_x_train.shape[0]].reshape(-1, 1),
            fashion_y_train[:boston_x_train.shape[0]].reshape(-1, 1),
            boston_y_train.reshape(-1, 1)
        ]
    )
    model.compile(
        loss = [
            'sparse_categorical_crossentropy',
            'sparse_categorical_crossentropy',
            'mse'
        ],
        optimizer = 'adam',
        loss_weights = [1, 1, 0]
    )
    model.fit(
        [digit_x_train, fashion_x_train, np.zeros((digit_x_train.shape[0], boston_x_train.shape[1]))],
        [digit_y_train, fashion_y_train, np.zeros(digit_x_train.shape[0])],
        batch_size = 512,
        epochs = 100,
        callbacks = [callback, mann_tboard],
        verbose = 0,
        validation_split = 0.2
    )
        
    model.compile(
        loss = [
            'sparse_categorical_crossentropy',
            'sparse_categorical_crossentropy',
            'mse'
        ],
        optimizer = 'adam',
        loss_weights = [0, 0, 1]
    )
    model.fit(
        [
            np.zeros((boston_x_train.shape[0], digit_x_train.shape[1])),
            np.zeros((boston_x_train.shape[0], fashion_x_train.shape[1])),
            boston_x_train
        ],
        [
            np.zeros(boston_x_train.shape[0]),
            np.zeros(boston_x_train.shape[0]),
            boston_y_train
        ],
        epochs = 100,
        callbacks = [callback, mann_tboard],
        batch_size = 32,
        verbose = 0,
        validation_split = 0.2
    )

    model = mann.utils.utils.remove_layer_masks(model)

    mnist_preds = model.predict(
        [digit_x_test, fashion_x_test, np.zeros((digit_x_test.shape[0], boston_x_test.shape[1]))]
    )
    digit_loss = tf.keras.losses.sparse_categorical_crossentropy(digit_y_test, mnist_preds[0])
    fashion_loss = tf.keras.losses.sparse_categorical_crossentropy(fashion_y_test, mnist_preds[1])
    digit_preds = mnist_preds[0].argmax(axis = 1)
    fashion_preds = mnist_preds[1].argmax(axis = 1)
    boston_preds = model.predict(
        [
            np.zeros((boston_x_test.shape[0], digit_x_test.shape[1])),
            np.zeros((boston_x_test.shape[0], fashion_x_test.shape[1])),
            boston_x_test
        ]
    )[2]
    model.save('multitask_model.h5')

    print('Multitask Model Digit Performance:')
    print(f'Loss: {digit_loss.numpy().mean()}')
    print(confusion_matrix(digit_y_test, digit_preds))
    print(classification_report(digit_y_test, digit_preds))
    print('\n')
    print('Multitask Model Fashion Performance:')
    print(f'Loss: {fashion_loss.numpy().mean()}')
    print(confusion_matrix(fashion_y_test, fashion_preds))
    print(classification_report(fashion_y_test, fashion_preds))
    print('\n')
    print(f'Multitask Model Boston Loss {tf.keras.losses.mse(boston_y_test, boston_preds).numpy().mean()}')

    digit_input = tf.keras.layers.Input(digit_x_train.shape[1:])
    fashion_input = tf.keras.layers.Input(fashion_x_train.shape[1:])
    boston_input = tf.keras.layers.Input(boston_x_train.shape[1:])
    boston_reshape = mann.layers.SparseDense.from_layer(model.layers[3])
    
    x = mann.layers.SparseMultiDense.from_layer(model.layers[4])
    for i in range(HIDDEN_LAYERS - 1):
        x = mann.layers.SparseMultiDense.from_layer(model.layers[i + 5])
        
    digit_selector = mann.layers.SelectorLayer(0)(x)
    fashion_selector = mann.layers.SelectorLayer(1)(x)
    boston_selector = mann.layers.SelectorLayer(2)(x)
    
    digit_output = mann.layers.SparseDense.from_layer(model.layers[-3])
    fashion_output = mann.layers.SparseDense.from_layer(model.layers[-2])
    boston_output = mann.layers.SparseDense.from_layer(model.layers[-1])

    model = tf.keras.models.Model([digit_input, fashion_input, boston_input], [digit_output, fashion_output, boston_output])

    print('Sparse Multitask Model Digit Performance:')
    print(f'Loss: {digit_loss.numpy().mean()}')
    print(confusion_matrix(digit_y_test, digit_preds))
    print(classification_report(digit_y_test, digit_preds))
    print('\n')
    print('Sparse Multitask Model Fashion Performance:')
    print(f'Loss: {fashion_loss.numpy().mean()}')
    print(confusion_matrix(fashion_y_test, fashion_preds))
    print(classification_report(fashion_y_test, fashion_preds))
    print('\n')
    print(f'Sparse Multitask Model Boston Loss {tf.keras.losses.mse(boston_y_test, boston_preds).numpy().mean()}')

    with open('sparse_model.pkl', 'wb') as f:
        pickle.dump(model, f)
