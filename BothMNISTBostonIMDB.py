from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import mann
import os

if __name__ == '__main__':

    HIDDEN_LAYERS = 6
    HIDDEN_NODES = 1000

    (digit_x_train, digit_y_train), (digit_x_test, digit_y_test) = tf.keras.datasets.mnist.load_data()
    digit_x_train = digit_x_train.reshape((digit_x_train.shape[0], -1))/255
    digit_x_test = digit_x_test.reshape((digit_x_test.shape[0], -1))/255
    digit_y_train = digit_y_train.reshape(-1, 1)
    digit_y_test = digit_y_test.reshape(-1, 1)

    (fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()
    fashion_x_train = fashion_x_train.reshape((fashion_x_train.shape[0], -1))/255
    fashion_x_test = fashion_x_test.reshape((fashion_x_test.shape[0], -1))/255
    fashion_y_train = fashion_y_train.reshape(-1, 1)
    fashion_y_test = fashion_y_test.reshape(-1, 1)

    (boston_x_train, boston_y_train), (boston_x_test, boston_y_test) = tf.keras.datasets.boston_housing.load_data()
    boston_x_scaler = MinMaxScaler().fit(boston_x_train)
    boston_y_scaler = MinMaxScaler().fit(boston_y_train.reshape(-1, 1))

    boston_x_train = boston_x_scaler.transform(boston_x_train)
    boston_x_test = boston_x_scaler.transform(boston_x_test)
    boston_y_train = boston_y_scaler.transform(boston_y_train.reshape(-1, 1))
    boston_y_test = boston_y_scaler.transform(boston_y_test.reshape(-1, 1))

    (imdb_x_train, imdb_y_train), (imdb_x_test, imdb_y_test) = tf.keras.datasets.imdb.load_data(num_words = 10000)

    imdb_x_train = tf.keras.preprocessing.sequence.pad_sequences(
        imdb_x_train,
        maxlen = 128,
        padding = 'post',
        truncating = 'post'
    )
    imdb_x_test = tf.keras.preprocessing.sequence.pad_sequences(
        imdb_x_test,
        maxlen = 128,
        padding = 'post',
        truncating = 'post'
    )

    imdb_y_train = imdb_y_train.reshape(-1, 1)
    imdb_y_test = imdb_y_test.reshape(-1, 1)

    callback = tf.keras.callbacks.EarlyStopping(
        min_delta = 0.01,
        patience = 3,
        restore_best_weights = True
    )

    log_dir = os.path.join('.', 'logs', 'BothMNISTBostonIMDBfc')
    imdb_log_dir = os.path.join(log_dir, 'IMDB')
    mann_log_dir = os.path.join(log_dir, 'mann')

    imdb_tboard = tf.keras.callbacks.TensorBoard(
        log_dir = imdb_log_dir,
        histogram_freq = 1
    )

    mann_tboard = tf.keras.callbacks.TensorBoard(
        log_dir = mann_log_dir,
        histogram_freq = 1
    )
    
    # Build the IMDB control model
    imdb_input = tf.keras.layers.Input(128)
    x = tf.keras.layers.Embedding(10000, 2)(imdb_input)
    x = tf.keras.layers.Flatten()(x)
    for _ in range(HIDDEN_LAYERS):
        x = tf.keras.layers.Dense(HIDDEN_NODES, activation = 'relu')(x)
    imdb_output = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

    model = tf.keras.models.Model(imdb_input, imdb_output)
    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
    model.fit(imdb_x_train, imdb_y_train, epochs = 100, batch_size = 512, validation_split = 0.2, callbacks = [callback, imdb_tboard])
    imdb_preds = (model.predict(imdb_x_test) >= 0.5).astype(int)
        
    print('IMDB Control Model Performance:')
    print(confusion_matrix(imdb_y_test, imdb_preds))
    print(classification_report(imdb_y_test, imdb_preds))

    # MANN Model
    digit_input = tf.keras.layers.Input(digit_x_train.shape[1:])
    fashion_input = tf.keras.layers.Input(fashion_x_train.shape[1:])
    boston_input = tf.keras.layers.Input(boston_x_train.shape[1:])
    imdb_input = tf.keras.layers.Input(imdb_x_train.shape[1:])

    boston_x = mann.layers.MaskedDense(HIDDEN_NODES, activation = 'relu')(boston_input)

    imdb_x = tf.keras.layers.Embedding(10000, 2)(imdb_input)
    imdb_x = tf.keras.layers.Flatten()(imdb_x)
    imdb_x = mann.layers.MaskedDense(HIDDEN_NODES, activation = 'relu')(imdb_x)

    image_x = mann.layers.MultiMaskedDense(HIDDEN_NODES, activation = 'relu')([digit_input, fashion_input])
    digit_x = mann.layers.SelectorLayer(0)(image_x)
    fashion_x = mann.layers.SelectorLayer(1)(image_x)

    mann_x = mann.layers.MultiMaskedDense(HIDDEN_NODES, activation = 'relu')([digit_x, fashion_x, boston_x, imdb_x])

    for _ in range(HIDDEN_LAYERS - 2):
        mann_x = mann.layers.MultiMaskedDense(HIDDEN_NODES, activation = 'relu')(mann_x)

    digit_x = mann.layers.SelectorLayer(0)(mann_x)
    fashion_x = mann.layers.SelectorLayer(1)(mann_x)
    boston_x = mann.layers.SelectorLayer(2)(mann_x)
    imdb_x = mann.layers.SelectorLayer(3)(mann_x)

    digit_output = mann.layers.MaskedDense(10, activation = 'softmax')(digit_x)
    fashion_output = mann.layers.MaskedDense(10, activation = 'softmax')(fashion_x)
    boston_output = mann.layers.MaskedDense(1, activation = 'relu')(boston_x)
    imdb_output = mann.layers.MaskedDense(1, activation = 'sigmoid')(imdb_x)

    model = tf.keras.models.Model(
        [digit_input, fashion_input, boston_input, imdb_input],
        [digit_output, fashion_output, boston_output, imdb_output]
    )
    model.compile(
        loss = ['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'mse', 'binary_crossentropy'],
        optimizer = 'adam'
    )
    model = mann.utils.mask_model(
        model,
        90,
        x = [
            digit_x_train[:boston_x_train.shape[0]],
            fashion_x_train[:boston_x_train.shape[0]],
            boston_x_train,
            imdb_x_train[:boston_x_train.shape[0]]
        ],
        y = [
            digit_y_train[:boston_x_train.shape[0]],
            fashion_y_train[:boston_x_train.shape[0]],
            boston_y_train,
            imdb_y_train[:boston_x_train.shape[0]]
        ]
    )
    model.compile(
        loss = ['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'mse', 'binary_crossentropy'],
        optimizer = 'adam',
        loss_weights = [1, 1, 0, 0]
    )

    # Train the model on the image tasks
    model.fit(
        [digit_x_train, fashion_x_train, np.zeros((digit_x_train.shape[0],boston_x_train.shape[1])), np.zeros((digit_x_train.shape[0], imdb_x_train.shape[1]))],
        [digit_y_train, fashion_y_train, np.zeros(digit_x_train.shape[0]).reshape(-1, 1), np.zeros(digit_x_train.shape[0]).reshape(-1 ,1)],
        epochs = 100,
        batch_size = 512,
        validation_split = 0.2,
        callbacks = [callback, mann_tboard]
    )

    model.compile(
        loss = ['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'mse', 'binary_crossentropy'],
        optimizer = 'adam',
        loss_weights = [0, 0, 1, 0]
    )
    
    # Train the model on the boston task
    model.fit(
        [np.zeros((boston_x_train.shape[0], digit_x_train.shape[1])), np.zeros((boston_x_train.shape[0], fashion_x_train.shape[1])), boston_x_train, np.zeros((boston_x_train.shape[0], imdb_x_train.shape[1]))],
        [np.zeros(boston_x_train.shape[0]).reshape(-1, 1), np.zeros(boston_x_train.shape[0]).reshape(-1, 1), boston_y_train, np.zeros(boston_x_train.shape[0]).reshape(-1, 1)],
        epochs = 100,
        batch_size = 32,
        validation_split = 0.2,
        callbacks = [callback, mann_tboard]
    )

    model.compile(
        loss = ['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'mse', 'binary_crossentropy'],
        optimizer = 'adam',
        loss_weights = [0, 0, 0, 1]
    )

    # Train the model on the IMDB task
    model.fit(
        [np.zeros((imdb_x_train.shape[0], digit_x_train.shape[1])), np.zeros((imdb_x_train.shape[0], fashion_x_train.shape[1])), np.zeros((imdb_x_train.shape[0], boston_x_train.shape[1])), imdb_x_train],
        [np.zeros(imdb_x_train.shape[0]).reshape(-1, 1), np.zeros(imdb_x_train.shape[0]).reshape(-1, 1), np.zeros(imdb_x_train.shape[0]).reshape(-1, 1), imdb_y_train],
        epochs = 100,
        batch_size = 32,
        validation_split = 0.2,
        callbacks = [callback, mann_tboard]
    )
