import beyondml.tflow as mann
import tensorflow as tf
import numpy as np

vocab_size = 30000
maxlen = 512
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(num_words = vocab_size)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen = maxlen)
x_train_positions = [np.arange(maxlen)]*x_train.shape[0]
x_train_positions = np.asarray(x_train.shape[0] * [np.arange(x_train.shape[1])])
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen = maxlen)
x_test_positions = np.asarray(x_test.shape[0] * [np.arange(x_test.shape[1])])
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(x_train.shape)
print(x_train_positions.shape)

embed_dim = 512
num_heads = 8
ff_dim = 1028

dropout = 0.1

token_input = tf.keras.layers.Input(maxlen)
pos_input = tf.keras.layers.Input(maxlen)
x = mann.utils.build_token_position_embedding_block(
    maxlen,
    vocab_size,
    embed_dim
)([token_input, pos_input])
x = mann.utils.build_transformer_block((maxlen, embed_dim), embed_dim, num_heads, ff_dim)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(dropout)(x)
x = tf.keras.layers.Dense(ff_dim, activation = 'relu')(x)
x = tf.keras.layers.Dropout(dropout)(x)
output_layer = tf.keras.layers.Dense(np.unique(y_train).shape[0], activation = 'softmax')(x)

model = tf.keras.models.Model([token_input, pos_input], output_layer)
model = mann.utils.add_layer_masks(model)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(
    min_delta = 0.004,
    patience = 3,
    restore_best_weights = True
)



# Fit the model
model.fit(
    [x_train, x_train_positions],
    y_train,
    batch_size = 256,
    epochs = 2,
    validation_split = 0.2,
    epochs = 100,
    callbacks = [callback]
)
model.summary()
