import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from tqdm import tqdm
import numpy as np

NUM_OF_CLASSES = 4251  # len(np.unique(y))# 4251

training_dir = 'data/train/'
data = pd.read_csv('data/train.csv')
data['Id_int'] = pd.factorize(data['Id'])[0]

train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=1337)
print('Checking training data head')
print(train.head())
print('Checking test data head')
print(test.head())

sampleImageFile1 = train.Image[2]
sampleImage = mpimg.imread(training_dir + sampleImageFile1)
plt.imshow(sampleImage)
plt.show()

vc = train.Id.value_counts().sort_values(ascending=False)
vc[:50].plot(kind='bar')
plt.show()


def process(image):
    # resize
    image = np.resize(image, [128, 128, 1])

    # convert to grayscale
    if image.shape == 3:
        image = np.dot([image[:,:,0],image[:,:,1],image[:,:,2]],[0.299,0.587,0.114])

    # return normalized
    return image / 255


def get_image_from_path(path, training_dir, data):
    image = mpimg.imread(training_dir + path)
    image = process(image)
    label = data[data.Image == path]['Id_int'].values[0]

    return image, label


def loop_over_data(data):
    X = []
    y = []

    for index, path in tqdm(enumerate(data.Image)):
        if index>10:
            break
        image, label = get_image_from_path(path, training_dir, data)

        X.append(image)
        y.append(label)

    return np.array(X), y


X_train, y_train = loop_over_data(train)
X_test, y_test = loop_over_data(test)


print('X_train: ', np.shape(X_train))
print('y_train: ', np.shape(y_train))

print('X_test: ', np.shape(X_test))
print('y_test: ', np.shape(y_test))

# One hot encoding
train_Y_one_hot = to_categorical(y_train, num_classes=NUM_OF_CLASSES)
test_Y_one_hot = to_categorical(y_test, num_classes=NUM_OF_CLASSES)
print('After conversion to one-hot:', len(train_Y_one_hot[0]))
print('shape train_Y_one_hot: ', np.shape(train_Y_one_hot))

k_size = (4, 4)
drop_probability = 0.5
hidden_size = 256
batch_size = 64
input_shape = (128, 128, 1)
pool_size = (2, 2)
learning_rate = 0.07
num_of_epochs = 3

print('num_of_classes: ', NUM_OF_CLASSES)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',
          input_shape=input_shape, padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(NUM_OF_CLASSES, activation='softmax'))


model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])
model.summary()

model.fit(X_train, train_Y_one_hot, batch_size=batch_size,
          epochs=num_of_epochs, verbose=1)

model.save('whale.h5py')
test_eval = model.evaluate(X_test, test_Y_one_hot, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
