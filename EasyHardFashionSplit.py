from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import numpy as np
import mann

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape((x_train.shape + (1,)))/255
    x_test = x_test.reshape((x_test.shape + (1,)))/255

    easy_values = [
        1,
        5,
        7,
        8,
        9
    ]
    hard_values = [
        val for val in range(10) if val not in easy_values
    ]
    
    easy_x_train = x_train[np.isin(y_train, easy_values)]
    easy_y_train = y_train[np.isin(y_train, easy_values)]
    hard_x_train = x_train[np.isin(y_train, hard_values)]
    hard_y_train = y_train[np.isin(y_train, hard_values)]

    easy_x_test = x_test[np.isin(y_test, easy_values)]
    easy_y_test = y_test[np.isin(y_test, easy_values)]
    hard_x_test = x_test[np.isin(y_test, hard_values)]
    hard_y_test = y_test[np.isin(y_test, hard_values)]

    easy_mapper = dict(zip(easy_values, range(len(easy_values))))
    hard_mapper = dict(zip(hard_values, range(len(hard_values))))

    easy_y_train = np.array(
        [easy_mapper[val] for val in easy_y_train.flatten()]
    )
    easy_y_test = np.array(
        [easy_mapper[val] for val in easy_y_test.flatten()]
    )
    hard_y_train = np.array(
        [hard_mapper[val] for val in hard_y_train.flatten()]
    )
    hard_y_test = np.array(
        [hard_mapper[val] for val in hard_y_test.flatten()]
    )

    callback = tf.keras.callbacks.EarlyStopping(
        min_delta = 0.01,
        patience = 3,
        restore_best_weights = True
    )
    
    input1 = tf.keras.layers.Input(easy_x_train.shape[1:])
    input2 = tf.keras.layers.Input(hard_x_train.shape[1:])
    x = mann.layers.MultiMaskedConv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )([input1, input2])
    x = mann.layers.MultiMaskedConv2D(
        32,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    sel1 = mann.layers.SelectorLayer(0)(x)
    sel2 = mann.layers.SelectorLayer(1)(x)
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
    x = mann.layers.MultiMaskedConv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )([pool1, pool2])
    x = mann.layers.MultiMaskedConv2D(
        64,
        3,
        padding = 'same',
        activation = 'relu'
    )(x)
    sel1 = mann.layers.SelectorLayer(0)(x)
    sel2 = mann.layers.SelectorLayer(1)(x)
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
    flat1 = tf.keras.layers.Flatten()(pool1)
    flat2 = tf.keras.layers.Flatten()(pool2)
    x = mann.layers.MultiMaskedDense(256, activation = 'relu')(
        [
            flat1,
            flat2
        ]
    )
    x = mann.layers.MultiMaskedDense(256, activation = 'relu')(x)
    output_layer = mann.layers.MultiMaskedDense(10, activation = 'softmax')(x)
    model = tf.keras.models.Model(
        [input1, input2],
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
        x = [easy_x_train[:1000], hard_x_train[:1000]],
        y = [easy_y_train[:1000].reshape(-1, 1), hard_y_train[:1000].reshape(-1, 1)]
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    model.fit(
        [easy_x_train, hard_x_train],
        [easy_y_train, hard_y_train],
        epochs = 100,
        batch_size = 512,
        callbacks = [callback],
        validation_split = 0.2,
        verbose = 0
    )

    model = mann.utils.utils.convert_model(model)

    easy_preds, hard_preds = model.predict([easy_x_test, hard_x_test])
    easy_preds = easy_preds.argmax(axis = 1)
    hard_preds = hard_preds.argmax(axis = 1)

    print('Easy Performance:')
    print(confusion_matrix(easy_y_test, easy_preds))
    print(classification_report(easy_y_test, easy_preds))
    print('\n')

    print('Hard Performance:')
    print(confusion_matrix(hard_y_test, hard_preds))
    print(classification_report(hard_y_test, hard_preds))
