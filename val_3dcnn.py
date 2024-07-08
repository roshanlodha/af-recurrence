import pickle
import numpy as np
import random
import scipy.ndimage as ndi
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn import metrics

def load_data():
    with open('/home/lodhar/af-recurrence/train_val_data.pkl', 'rb') as file:
        dataset_dict = pickle.load(file)
    return dataset_dict['x_train'], dataset_dict['y_train'], dataset_dict['x_val'], dataset_dict['y_val']

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

    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def plot_auc(y_val, y_pred, best_auc):
    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
    plt.plot(fpr, tpr, linestyle='--', label='No Skill')
    plt.title("Best Model AUC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.text(0.6, 0, 'AUC = %s' % best_auc)
    plt.savefig("/home/lodhar/af-recurrence/figs/model_AUC.png")

def main():
    x_train, y_train, x_val, y_val = load_data()

    # Define data loaders.
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    batch_size = 1

    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )

    # Build model.
    model = get_model(width=256, height=256, depth=128)
    model.summary()

    # Load model weights.
    model.load_weights("/home/lodhar/af-recurrence/04-0.92.h5")

    # Predict and evaluate.
    y_pred = model.predict(validation_dataset).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
    best_auc = metrics.auc(fpr, tpr)

    # Plot AUC.
    plot_auc(y_val, y_pred, best_auc)

if __name__ == "__main__":
    main()
