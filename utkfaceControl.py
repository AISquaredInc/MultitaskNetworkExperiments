from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import click
import mann
import os

def data_generator(
        utkface_dir,
        task,
        batch_size,
        image_size = (256, 256),
        scaling = 1./255
):
    files = os.listdir(utkface_dir)
    np.random.shuffle(files)

    cutoffs = list(range(10, 100, 10))

    utkface_idx = 0

    while True:
        utkface_batch = []
        labels = []
        
        for _ in range(batch_size):
            if utkface_idx >= len(files):
                np.random.shuffle(files)
                utkface_idx = 0
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(utkface_dir, files[utkface_idx]),
                target_size = image_size
            )
            utkface_img = np.array(img)*scaling
            if task == 'age':
                age = int(files[utkface_idx].split('_')[0])
                age_label = sum([age > cutoff for cutoff in cutoffs])
                labels.append(age_label)
            elif task == 'gender':
                gender = int(files[utkface_idx].split('_')[1])
                labels.append(gender)
            elif task == 'ethnicity':
                ethnicity = int(files[utkface_idx].split('_')[2])
                labels.append(ethnicity)
                
            utkface_batch.append(utkface_img)
            utkface_idx += 1

        yield np.asarray(utkface_batch), np.asarray(labels)

def build_model(output_shape):
    input_layer = tf.keras.layers.Input((256, 256, 3))
    x = tf.keras.layers.Conv2D(
        16,
        3,
        activation = 'relu',
        padding = 'same'
    )(input_layer)
    x = tf.keras.layers.Conv2D(
        16,
        3,
        activation = 'relu',
        padding = 'same'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(
        32,
        3,
        activation = 'relu',
        padding = 'same'
    )(x)
    x = tf.keras.layers.Conv2D(
        32,
        3,
        activation = 'relu',
        padding = 'same'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(
        64,
        3,
        activation = 'relu',
        padding = 'same'
    )(x)
    x = tf.keras.layers.Conv2D(
        64,
        3,
        activation = 'relu',
        padding = 'same'
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(
        128,
        3,
        activation = 'relu',
        padding = 'same'
    )(x)
    x = tf.keras.layers.Conv2D(
        128,
        3,
        activation = 'relu',
        padding = 'same'
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
@click.option('--limit', '-l', type = int, default = None)
def main(train_dir, val_dir, batch_size, limit):
    age_model = build_model(10)
    gender_model = build_model(1)
    ethnicity_model = build_model(5)

    age_train_generator = data_generator(train_dir, 'age', batch_size)
    gender_train_generator = data_generator(train_dir, 'gender', batch_size)
    ethnicity_train_generator = data_generator(train_dir, 'ethnicity', batch_size)

    age_val_generator = data_generator(val_dir, 'age', batch_size)
    gender_val_generator = data_generator(val_dir, 'gender', batch_size)
    ethnicity_val_generator = data_generator(val_dir, 'ethnicity', batch_size)

    if not limit:
        train_steps = len(os.listdir(train_dir))//batch_size
        val_steps = len(os.listdir(val_dir))//batch_size
    else:
        train_steps = limit
        val_steps = limit

    callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0.01,
        patience = 5,
        restore_best_weights = True
    )

    age_model.fit(
        age_train_generator,
        epochs = 100,
        steps_per_epoch = train_steps,
        validation_data = age_val_generator,
        validation_steps = val_steps,
        callbacks = [callback]
    )

    gender_model.fit(
        gender_train_generator,
        epochs = 100,
        steps_per_epoch = train_steps,
        validation_data = gender_val_generator,
        validation_steps = val_steps,
        callbacks = [callback]
    )

    ethnicity_model.fit(
        ethnicity_train_generator,
        epochs = 100,
        steps_per_epoch = train_steps,
        validation_data = ethnicity_val_generator,
        validation_steps = val_steps,
        callbacks = [callback]
    )

    # get predictions from all models
    age_preds, age_labels = [], []
    gender_preds, gender_labels = [], []
    ethnicity_preds, ethnicity_labels = [], []
    
    for _ in range(val_steps):
        age_data, age_labs = next(age_val_generator)
        age_p = age_model.predict(age_data)

        gender_data, gender_labs = next(gender_val_generator)
        gender_p = gender_model.predict(gender_data)

        ethnicity_data, ethnicity_labs = next(ethnicity_val_generator)
        ethnicity_p = ethnicity_model.predict(ethnicity_data)

        age_preds.extend(age_p.argmax(axis = 1).flatten().tolist())
        gender_preds.extend((gender_p >= 0.5).astype(int).flatten().tolist())
        ethnicity_preds.extend(ethnicity_p.argmax(axis = 1).flatten().tolist())

        age_labels.extend(age_labs.flatten().tolist())
        gender_labels.extend(gender_labs.flatten().tolist())
        ethnicity_labels.extend(ethnicity_labs.flatten().tolist())
        
    age_preds = np.asarray(age_preds)
    gender_preds = np.asarray(gender_preds)
    ethnicity_preds = np.asarray(ethnicity_preds)

    age_labels = np.asarray(age_labels)
    gender_labels = np.asarray(gender_labels)
    ethnicity_labels = np.asarray(ethnicity_labels)

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

    age_model.save('age_model.h5')
    gender_model.save('gender_model.h5')
    ethnicity_model.save('ethnicity_model.h5')

if __name__ == '__main__':
    main()
