# Import all the necessary libraries
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 128 neurons
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 128 neurons
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # 10 neurons in output layer

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # train the model
# model.fit(x_train, y_train, epochs=100)
#
# # Save model
# model.save('HWDR.model')
#
# # evaluate loss and accuracy
# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss * 100)
# print(accuracy * 100)

#load model
model = tf.keras.models.load_model('HWDR.model')

# Make predictions
image_number = 1
while os.path.isfile(f"image/Untitled{image_number}.png"):
    try:
        img = cv2.imread(f"image/Untitled{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("the result is probably: {}".format(np.argmax(prediction)))  # print the result
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("An exception occurred")
    finally:
        image_number += 1
