import pickle
import numpy as np
import random
import scipy.ndimage as ndi
import tensorflow as tf
import keras
from keras.utils import plot_model
from keras import layers
import matplotlib.pyplot as plt

def load_data():
    with open('/home/lodhar/af-recurrence/train_val_data.pkl', 'rb') as file:
        dataset_dict = pickle.load(file)
    return dataset_dict['x_train'], dataset_dict['y_train'], dataset_dict['x_val'], dataset_dict['y_val']

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""
    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndi.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    volume = rotate(volume) # -- in theory the volumes shouldnt NEED to be rotated...
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def get_model(width=256, height=256, depth=128):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(["accuracy", "loss"]):
        ax[i].plot(history.history[metric])
        ax[i].plot(history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])

    plt.savefig('/home/lodhar/af-recurrence/figs/model_loss.png')

def main():
    x_train, y_train, x_val, y_val = load_data()

    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    batch_size = 4
    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(4)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(4)
    )

    # Build model.
    model = get_model(width=256, height=256, depth=128)
    model.summary()

    # Compile model.
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["accuracy", tf.keras.metrics.AUC(name='auc')],
    )

    plot_model(model, to_file='/home/lodhar/af-recurrence/figs/model_architecture.png', show_shapes=True, show_dtype=False, show_layer_names=False, rankdir='TB', show_layer_activations=True, expand_nested=True, show_trainable=False)

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "{epoch:02d}-{val_loss:.2f}.h5", 
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        #initial_value_threshold=0.65,
        verbose=1
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_auc", patience=25, mode='max')

    # Train the model, doing validation at the end of each epoch
    epochs = 100
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    plot_history(history)

if __name__ == "__main__":
    main()
