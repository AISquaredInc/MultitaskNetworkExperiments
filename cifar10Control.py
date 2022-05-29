from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import click
import mann
import os

def build_model(output_shape):
    input_layer = tf.keras.layers.Input((256, 256, 3))
    x = tf.keras.layers.Conv2D(
        16,
        3,
        activation = 'relu'
    )(input_layer)
    x = tf.keras.layers.Conv2D(
        16,
        3,
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(
        32,
        3,
        activation = 'relu'
    )(x)
    x = tf.keras.layers.Conv2D(
        32,
        3,
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(
        64,
        3,
        activation = 'relu'
    )(x)
    x = tf.keras.layers.Conv2D(
        64,
        3,
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(
        128,
        3,
        activation = 'relu'
    )(x)
    x = tf.keras.layers.Conv2D(
        128,
        3,
        activation = 'relu'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    for _ in range(3):
        x = tf.keras.layers.Dense(128, activation = 'relu')(x)
    output_layer = tf.keras.layers.Dense(output_shape)(x)

    if output_shape == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    
    model = tf.keras.models.Model(input_layer, output_layer)
    model.compile(
        loss = loss,
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    return model
        
@click.command()
@click.argument('train-dir', type = click.Path(exists = True, dir_okay = True, file_okay = False))
@click.argument('val-dir', type = click.Path(exists = True, dir_okay = True, file_okay = False))
@click.option('--batch-size', '-b', type = int, default = 256)
def main(train_dir, val_dir, batch_size, limit):

    model = build_model(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = tf.keras.image.resize(x_train, (256, 256))/255
    x_test = tf.keras.image.resize(x_test, (256, 256))/255
    
    callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0.01,
        patience = 5,
        restore_best_weights = True
    )

    model.fit(
        x_train,
        y_train,
        batch_size = batch_size,
        epochs = 100,
        validation_split = 0.2,
        callbacks = [callback]
    )

    preds = model.predict(x_test).argmax(axis = 1)

    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    model.save('cifar10Control.py')

if __name__ == '__main__':
    main()
