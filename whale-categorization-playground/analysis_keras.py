import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from tqdm import tqdm
import numpy as np

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
    image = np.resize(image, [128, 128])

    # convert to grayscale
    if image.shape == 3:
        image = np.dot([image[:,:,0],image[:,:,1],image[:,:,2]],[0.299,0.587,0.114])

    # return normalized
    return image / 255

x = []
y = []




for index, path in tqdm(enumerate(train.Image)):
    if index>4:
        break
    image = mpimg.imread(training_dir + path)
    image = process(image)
    x.append(image)

    cod = train[train.Image == path]['Id_int'].values[0]
    y.append(cod)

print(np.shape(x))
y
print(np.shape(y))


k_size = (4, 4)
drop_probability = 0.5
hidden_size = 256
batch_size = 64
input_shape = (batch_size, 128, 128)
pool_size = (2,2)
learning_rate = 0.07
num_of_epochs = 1
num_of_classes = 4251

model = Sequential()
model.add(Convolution2D(32, kernel_size=k_size, activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2,2)))
model.add(Convolution2D(64, kernel_size=k_size, activation="relu"))
model.add(MaxPooling2D(pool_size=pool_size, strides=(1,1)))
model.add(Convolution2D(512, kernel_size=k_size, activation="relu"))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2,2)))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(512, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(num_of_classes, activation="softmax"))

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])


model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=num_of_epochs, verbose=1)
