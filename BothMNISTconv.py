from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import mann

if __name__ == '__main__':

    (digit_x_train, digit_y_train), (digit_x_test, digit_y_test) = tf.keras.datasets.mnist.load_data()
    digit_x_train = digit_x_train.reshape((digit_x_train.shape + (1,)))/255
    digit_x_test = digit_x_test.reshape((digit_x_test.shape + (1,)))/255
    
    (fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()
    fashion_x_train = fashion_x_train.reshape((fashion_x_train.shape + (1,)))/255
    fashion_x_test = fashion_x_test.reshape((fashion_x_test.shape + (1,)))/255

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
        callbacks = [callback]
    )
    digit_preds = model.predict(digit_x_test).argmax(axis = 1)
    
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
        callbacks = [callback]
    )
    fashion_preds = model.predict(fashion_x_test).argmax(axis = 1)

    print('Dedicated Model Fashion Performance:')
    print(confusion_matrix(fashion_y_test, fashion_preds))
    print(classification_report(fashion_y_test, fashion_preds))
    
    print('Multitask Model Digit Performance:')
    print(confusion_matrix(digit_y_test, digit_preds))
    print(classification_report(digit_y_test, digit_preds))
    print('\n')
    print('Multitask Model Fashion Performance:')
    print(confusion_matrix(fashion_y_test, fashion_preds))
    print(classification_report(fashion_y_test, fashion_preds))
