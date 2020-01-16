import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, MaxPooling2D, Convolution2D as Conv2D
import json
from skimage.transform import resize


# Creates a CNN model
def create_model():
    input_shape = (80, 80, 3)
    numberOfCategories = 3

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Add Dropout layer to reduce overfitting

    model.add(Conv2D(32, (10, 10), activation='relu'))
    model.add(Dropout(0.25))  # Add Dropout layer to reduce overfitting

    model.add(Flatten())

    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=numberOfCategories, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    batch_size = 32
    epochs = 10
    verbose = 2

    return model, batch_size, epochs, verbose


# Train model on data
def evaluate(model, batch, num_epochs, verb, x_train, x_test, y_train, y_test):
    y_train = tf.keras.utils.to_categorical(y_train, 3)
    y_test = tf.keras.utils.to_categorical(y_test, 3)

    results = model.fit(x_train, y_train, batch_size=batch, epochs=num_epochs, verbose=verb,
                        validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=verb)

    print("Test accuracy: ", test_acc)

def main():
    np.random.seed(123)
    # Load Dataset
    file = open(r'shipsnet.json')
    dataset = json.load(file)
    images = []

    for index in range(len(dataset['data'])):
        pixel_vals = dataset['data'][index]
        arr = np.array(pixel_vals).astype('uint8')
        im = arr.reshape((3, 80 * 80)).T.reshape((80, 80, 3))
        ims = resize(im, (80, 80), mode='constant')
        images.append(ims)

    images = np.array(images)
    labels = np.array(dataset['labels']).astype('uint8')

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.975, shuffle="false")

    model, batch, num_epochs, verb = create_model()
    evaluate(model, batch, num_epochs, verb, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()

