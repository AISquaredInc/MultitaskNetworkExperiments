from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import mann
import cv2

if __name__ == '__main__':
    (cifar_x_train, cifar_y_train), (cifar_x_test, cifar_y_test) = tf.keras.datasets.cifar10.load_data()
    (digit_x_train, digit_y_train), (digit_x_test, digit_y_test) = tf.keras.datasets.mnist.load_data()
    (fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()

    callback = tf.keras.callbacks.EarlyStopping(
        min_delta = 0.01,
        patience = 3,
        restore_best_weights = True
    )
    
    cifar_x_train = np.array([cv2.resize(cifar_x_train[i], (28, 28)) for i in tqdm(range(cifar_x_train.shape[0]))])
    cifar_x_test = np.array([cv2.resize(cifar_x_test[i], (28, 28)) for i in tqdm(range(cifar_x_test.shape[0]))])
    digit_x_train = digit_x_train.reshape((digit_x_train.shape[0], 28, 28, 1))
    digit_x_test = digit_x_test.reshape((digit_x_test.shape[0], 28, 28, 1))
    fashion_x_train = fashion_x_train.reshape((fashion_x_train.shape[0], 28, 28, 1))
    fashion_x_test = fashion_x_test.reshape((fashion_x_test.shape[0], 28, 28, 1))

    input_layer = tf.keras.layers.Input(cifar_x_train.shape[1:])
    x = tf.keras.layers.Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = 'relu')(input_layer)
    x = tf.keras.layers.Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(x)
    x = tf.keras.layers.Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size = 2,
        strides = 1,
        padding = 'valid'
    )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(10, activation = 'softmax')(x)

    model = tf.keras.models.Model(input_layer, output_layer)
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    model.fit(
        cifar_x_train,
        cifar_y_train,
        epochs = 100,
        batch_size = 512,
        callbacks = [callback],
        validation_split = 0.2,
        verbose = 0
    )
    cifar_preds = model.predict(cifar_x_test).argmax(axis = 1)
    print('Dedicated Model CIFAR Performance:')
    print(confusion_matrix(cifar_y_test, cifar_preds))
    print(classification_report(cifar_y_test, cifar_preds))
    print('\n')
    
    cifar_input = tf.keras.layers.Input(cifar_x_train.shape[1:])
    digit_input = tf.keras.layers.Input(digit_x_train.shape[1:])
    fashion_input = tf.keras.layers.Input(fashion_x_train.shape[1:])
    
    cifar_conv = mann.layers.MaskedConv2D(32, padding = 'same', activation = 'relu')(cifar_input)
    mnist_conv = mann.layers.MultiMaskedConv2D(32, padding = 'same', activation = 'relu')([digit_input, fashion_input])
    x = mann.layers.MultiMaskedConv2D(32, padding = 'same', activation = 'relu')([cifar_conv] + mnist_conv)
    sel1 = mann.layers.SelectorLayer(0)(x)
    sel2 = mann.layers.SelectorLayer(1)(x)
    sel3 = mann.layers.SelectorLayer(2)(x)
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
    x = mann.layers.MultiMaskedConv2D(64, padding = 'same', activation = 'relu')([pool1, pool2, pool3])
    x = mann.layers.MultiMaskedConv2D(64, padding = 'same', activation = 'relu')(x)
    sel1 = mann.layers.SelectorLayer(0)(x)
    sel2 = mann.layers.SelectorLayer(1)(x)
    sel3 = mann.layers.SelectorLayer(2)(x)
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
    flat1 = tf.keras.layers.Flatten()(pool1)
    flat2 = tf.keras.layers.Flatten()(pool2)
    flat3 = tf.keras.layers.Flatten()(pool3)
    x = mann.layers.MultiMaskedDense(256, activation = 'relu')(
        [
            flat1,
            flat2,
            flat3
        ]
    )
    x = mann.layers.MultiMaskedDense(256, activation = 'relu')(x)
    output_layer = mann.layers.MultiMaskedDense(10, activation = 'softmax')(x)
    model = tf.keras.models.Model(
        [cifar_input, digit_input, fashion_input],
        output_layer
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    model = mann.utils.mask_model(
        model,
        90,
        method = 'gradients',
        x = [cifar_x_train[:10000], digit_x_train[:10000], fashion_x_train[:10000]],
        y = [cifar_y_train[:10000], digit_y_train[:10000], fashion_y_train[:10000]]
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam',
        loss_weights = [1, 0, 0]
    )
    
    model.fit(
        [cifar_x_train, digit_x_train[:cifar_x_train.shape[0]], fashion_x_train[:cifar_x_train.shape[0]]],
         [cifar_y_train, digit_y_train[:cifar_x_train.shape[0]], fashion_y_train[:cifar_x_train.shape[0]]],
        epochs = 100,
        batch_size = 512,
        callbacks = [callback],
        validation_split = 0.2,
        verbose = 0
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam',
        loss_weights = [0, 1, 1]
    )
    model.fit(
        [np.zeros((60000,) + cifar_x_train.shape[1:]), digit_x_train, fashion_x_train],
        [np.zeros(60000), digit_y_train, fashion_y_train],
        epochs = 100,
        batch_size = 512,
        callbacks = [callback],
        validation_split = 0.2,
        verbose = 0
    )

    model = mann.utils.utils.convert_model(model)

    cifar_preds = model.predict(
        [cifar_x_test, digit_x_test[:cifar_x_test.shape[0]], fashion_x_test[:cifar_x_test.shape[0]]],
        )[0].argmax(axis = 1)
    digit_preds, fashion_preds = model.predict(
        [np.zeros((10000,) + cifar_x_test.shape[1:]), digit_x_test, fashion_x_test]
    )[-2:]
    digit_preds = digit_preds.argmax(axis = 1)
    fashion_preds = fashion_preds.argmax(axis = 1)

    print('Multitask Model CIFAR Performance:')
    print(confusion_matrix(cifar_y_test, cifar_preds))
    print(classification_report(cifar_y_test, cifar_preds))
    print('\n')
    
    print('Multitask Model Digit Performance:')
    print(confusion_matrix(digit_y_test, digit_preds))
    print(classification_report(digit_y_test, digit_preds))

    print('Multitask Model Fashion Performance:')
    print(confusion_matrix(fashion_y_test, fashion_preds))
    print(classification_report(fashion_y_test, fashion_preds))
