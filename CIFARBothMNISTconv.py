from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import cv2

if __name__ == '__main__':
    (cifar_x_train, cifar_y_train), (cifar_x_test, cifar_y_test) = tf.keras.datasets.cifar10.load_data()
    (digit_x_train, digit_y_train), (digit_x_test, digit_y_test) = tf.keras.datasets.mnist.load_data()
    (fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()

    cifar_x_train = np.array([cv2.resize(cifar_x_train[i], (28, 28)) for i in tqdm(range(cifar_x_train.shape[0]))])
    cifar_x_test = np.array([cv2.resize(cifar_x_test[i], (28, 28)) for i in tqdm(range(cifar_x_test.shape[0]))])
    digit_x_train = digit_x_train.reshape((digit_x_train.shape[0], 28, 28, 1))
    digit_x_test = digit_x_test.reshape((digit_x_test.shape[0], 28, 28, 1))
    fashion_x_train = fashion_x_train.reshape((fashion_x_train.shape[0], 28, 28, 1))
    fashion_x_test = fashion_x_test.reshape((fashion_x_test.shape[0], 28, 28, 1))
    print(cifar_x_train.shape)
    print(cifar_x_test.shape)
    print(digit_x_train.shape)
