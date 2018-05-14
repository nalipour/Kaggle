from datetime import datetime

from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import helper
import visualization


%matplotlib inline

MAX_IMAGES = 10
now = datetime.utcnow().strftime('%Y%m%d%H%M')

training_dir = 'data/train/'
data, all_classes = helper.read_classes('data/train.csv')
num_of_classes = len(all_classes)

train, test = train_test_split(
    data, test_size=0.3, shuffle=True, random_state=1337)

visualization.count_individual(train)
visualization.view_image_index(data, training_dir, 50)
visualization.view_image_whaleID(data, training_dir, 'w_e6ec8ee')
# print('Checking training data head')
# print(train.head())
# print('Checking test data head')
# print(test.head())

X_train, y_train = helper.load_train_data(train, training_dir, max_images=MAX_IMAGES)
X_test, y_test = helper.load_train_data(test, training_dir, max_images=MAX_IMAGES)

# print('X_train: ', np.shape(X_train))
# print('y_train: ', np.shape(y_train))
#
# print('X_test: ', np.shape(X_test))
# print('y_test: ', np.shape(y_test))

# One hot encoding
train_Y_one_hot = to_categorical(y_train, num_classes=num_of_classes)
test_Y_one_hot = to_categorical(y_test, num_classes=num_of_classes)
# print('After conversion to one-hot:', len(train_Y_one_hot[0]))
# print('shape train_Y_one_hot: ', np.shape(train_Y_one_hot))

k_size = (4, 4)
drop_probability = 0.5
hidden_size = 256
batch_size = 64
input_shape = (128, 128, 1)
pool_size = (2, 2)
learning_rate = 0.07
num_of_epochs = 3

print('num_of_classes: ', num_of_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
                 input_shape=input_shape, padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_of_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])
model.summary()

whale_model = model.fit(X_train, train_Y_one_hot, batch_size=batch_size,
                        epochs=num_of_epochs, verbose=1)

model.save(now+'_whale.h5')
test_eval = model.evaluate(X_test, test_Y_one_hot, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

visualization.plot_accuracy_loss(whale_model)

input()
