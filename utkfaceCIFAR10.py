from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import click
import beyondml.tflow as mann
import os

def data_generator(
        utkface_dir,
        cifar10_images,
        cifar10_labels,
        batch_size,
        image_size = (256, 256),
        scaling = 1./255
):
    files = os.listdir(utkface_dir)
    np.random.shuffle(files)

    cifar10_images = tf.image.resize(cifar10_images, image_size)
    cifar10_images = cifar10_images*scaling

    cutoffs = list(range(10, 100, 10))

    utkface_idx = 0
    cifar10_idx = 0

    while True:
        utkface_batch = []
        cifar10_batch = []
        ages = []
        ethnicities = []
        genders = []
        cifar10_batch_labels = []

        for _ in range(batch_size):
            if utkface_idx >= len(files):
                np.random.shuffle(files)
                utkface_idx = 0
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(utkface_dir, files[utkface_idx]),
                target_size = image_size
            )
            utkface_img = np.array(img)*scaling
            age = int(files[utkface_idx].split('_')[0])
            age_label = sum([age > cutoff for cutoff in cutoffs])
            gender = int(files[utkface_idx].split('_')[1])
            ethnicity = int(files[utkface_idx].split('_')[2])
            utkface_batch.append(utkface_img)
            ages.append(age_label)
            genders.append(gender)
            ethnicities.append(ethnicity)
            

            if cifar10_idx >= cifar10_images.shape[0]:
                cifar10_idx = 0
            cifar10_batch.append(cifar10_images[cifar10_idx])
            cifar10_batch_labels.append(cifar10_labels[cifar10_idx])

            utkface_idx += 1
            cifar10_idx += 1

        yield ([np.asarray(utkface_batch), np.asarray(cifar10_batch)], [np.asarray(ages), np.asarray(genders), np.asarray(ethnicities), np.asarray(cifar10_batch_labels)])

def build_model():
    utkface_input = tf.keras.layers.Input((256, 256, 3))
    cifar10_input = tf.keras.layers.Input((256, 256, 3))

    x = mann.layers.MultiMaskedConv2D(
        16,
        activation = 'relu'
    )([utkface_input, cifar10_input])
    x = mann.layers.MultiMaskedConv2D(
        16,
        activation = 'relu'
    )(x)
    utkface_sel = mann.layers.SelectorLayer(0)(x)
    cifar10_sel = mann.layers.SelectorLayer(1)(x)
    utkface_pool = tf.keras.layers.MaxPool2D()(utkface_sel)
    cifar10_pool = tf.keras.layers.MaxPool2D()(cifar10_sel)
    x = mann.layers.MultiMaskedConv2D(
        32,
        activation = 'relu'
    )([utkface_pool, cifar10_pool])
    x = mann.layers.MultiMaskedConv2D(
        32,
        activation = 'relu'
    )(x)
    utkface_sel = mann.layers.SelectorLayer(0)(x)
    cifar10_sel = mann.layers.SelectorLayer(1)(x)
    utkface_pool = tf.keras.layers.MaxPool2D()(utkface_sel)
    cifar10_pool = tf.keras.layers.MaxPool2D()(cifar10_sel)
    x = mann.layers.MultiMaskedConv2D(
        64,
        activation = 'relu'
    )([utkface_pool, cifar10_pool])
    x = mann.layers.MultiMaskedConv2D(
        64,
        activation = 'relu'
    )(x)
    utkface_sel = mann.layers.SelectorLayer(0)(x)
    cifar10_sel = mann.layers.SelectorLayer(1)(x)
    utkface_pool = tf.keras.layers.MaxPool2D()(utkface_sel)
    cifar10_pool = tf.keras.layers.MaxPool2D()(cifar10_sel)
    utkface_flatten = tf.keras.layers.Flatten()(utkface_pool)
    cifar10_flatten = tf.keras.layers.Flatten()(cifar10_pool)

    x = mann.layers.MultiMaskedDense(
        128,
        activation = 'relu'
    )([utkface_flatten, utkface_flatten, utkface_flatten, cifar10_flatten])
    x = mann.layers.MultiMaskedDense(
        128,
        activation = 'relu'
    )(x)
    x = mann.layers.MultiMaskedDense(
        128,
        activation = 'relu'
    )(x)
    age_selector = mann.layers.SelectorLayer(0)(x)
    gender_selector = mann.layers.SelectorLayer(1)(x)
    ethnicity_selector = mann.layers.SelectorLayer(2)(x)
    cifar10_selector = mann.layers.SelectorLayer(3)(x)

    age_output = mann.layers.MaskedDense(10, activation = 'softmax')(age_selector)
    gender_output = mann.layers.MaskedDense(1, activation = 'sigmoid')(gender_selector)
    ethnicity_output = mann.layers.MaskedDense(5, activation = 'softmax')(ethnicity_selector)
    cifar10_output = mann.layers.MaskedDense(10, activation = 'softmax')(cifar10_selector)
    model = tf.keras.models.Model(
        [utkface_input, cifar10_input],
        [age_output, gender_output, ethnicity_output, cifar10_output]
    )
    model.compile(
        loss = [
            'sparse_categorical_crossentropy',
            'binary_crossentropy',
            'sparse_categorical_crossentropy',
            'sparse_categorical_crossentropy'
        ],
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    return model
        
@click.command()
@click.argument('train-dir', type = click.Path(exists = True, dir_okay = True, file_okay = False))
@click.argument('val-dir', type = click.Path(exists = True, dir_okay = True, file_okay = False))
@click.option('--batch-size', '-b', type = int, default = 256)
@click.option('--limit', '-l', type = int, default = None)
def main(train_dir, val_dir, batch_size, limit):
    (cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = tf.keras.datasets.cifar10.load_data()
    model = build_model()

    train_generator = data_generator(train_dir, cifar10_x_train, cifar10_y_train, batch_size)
    val_generator = data_generator(val_dir, cifar10_x_test, cifar10_y_test, batch_size)
    if not limit:
        train_steps = len(os.listdir(train_dir))//batch_size
        val_steps = len(os.listdir(val_dir))//batch_size
    else:
        train_steps = limit
        val_steps = limit

    model.fit(
        train_generator,
        epochs = 1,
        steps_per_epoch = train_steps,
        validation_data = val_generator,
        validation_steps = val_steps
    )

    model = mann.utils.mask_model(
        model,
        75,
        method = 'magnitude'
    )
    model.compile(
        loss = [
            'sparse_categorical_crossentropy',
            'binary_crossentropy',
            'sparse_categorical_crossentropy',
            'sparse_categorical_crossentropy'
        ],
        metrics = ['accuracy'],
        optimizer = 'adam'
    )
    
    callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0.01,
        patience = 5,
        restore_best_weights = True
    )
    
    model.fit(
        train_generator,
        epochs = 100,
        steps_per_epoch = train_steps,
        validation_data = val_generator,
        validation_steps = val_steps,
        callbacks = [callback]
    )

    model = mann.utils.remove_layer_masks(model)
    model.save('utkfaceAndCIFAR10.h5')
    
    age_preds = []
    gender_preds = []
    ethnicity_preds = []
    cifar_preds = []

    age_labels = []
    gender_labels = []
    ethnicity_labels = []
    cifar_labels = []
    
    for _ in range(val_steps):
        new_data, new_labels = next(val_generator)
        new_preds = model.predict(new_data)

        age_preds.extend(new_preds[0].argmax(axis = 1).flatten().tolist())
        gender_preds.extend((new_preds[1] >= 0.5).astype(int).flatten().tolist())
        ethnicity_preds.extend(new_preds[2].argmax(axis = 1).flatten().tolist())
        cifar_preds.extend(new_preds[3].argmax(axis = 1).flatten().tolist())

        age_labels.extend(new_labels[0].flatten().tolist())
        gender_labels.extend(new_labels[1].flatten().tolist())
        ethnicity_labels.extend(new_labels[2].flatten().tolist())
        cifar_labels.extend(new_labels[3].flatten().tolist())
        
    age_preds = np.asarray(age_preds)
    gender_preds = np.asarray(gender_preds)
    ethnicity_preds = np.asarray(ethnicity_preds)
    cifar_preds = np.asarray(cifar_preds)

    age_labels = np.asarray(age_labels)
    gender_labels = np.asarray(gender_labels)
    ethnicity_labels = np.asarray(ethnicity_labels)
    cifar_labels = np.asarray(cifar_labels)

    print('Age:')
    print(confusion_matrix(age_labels, age_preds))
    print(classification_report(age_labels, age_preds))
    print('\n\n')
    print('Gender:')
    print(confusion_matrix(gender_labels, gender_preds))
    print(classification_report(gender_labels, gender_preds))
    print('\n\n')
    print('Ethnicity:')
    print(confusion_matrix(ethnicity_labels, ethnicity_preds))
    print(classification_report(ethnicity_labels, ethnicity_preds))
    print('\n\n')
    print('CIFAR10:')
    print(confusion_matrix(cifar_labels, cifar_preds))
    print(classification_report(cifar_labels, cifar_preds))

if __name__ == '__main__':
    main()
