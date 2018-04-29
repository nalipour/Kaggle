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
    image = np.resize(image, [128, 128, 3])

    # convert to grayscale
    if image.shape == 3:
        image = np.dot([image[:,:,0],image[:,:,1],image[:,:,2]],[0.299,0.587,0.114])

    # return normalized
    return image / 255

x = []
y = []




for index, path in tqdm(enumerate(train.Image)):
    """
    if index>4:
        break
    """

    image = mpimg.imread(training_dir + path)
    image = process(image)
    x.append(image)

    cod = train[train.Image == path]['Id_int'].values[0]
    y.append(cod)


print(np.shape(x))
print(np.shape(y))
y
#x = x.astype('float32')

# One hot encoding
train_Y_one_hot = to_categorical(y)
print('After conversion to one-hot:', len(train_Y_one_hot[0]))
print('shape train_Y_one_hot: ', np.shape(train_Y_one_hot))

k_size = (4, 4)
drop_probability = 0.5
hidden_size = 256
batch_size = 64
input_shape = (128, 128, 3)
pool_size = (2,2)
learning_rate = 0.07
num_of_epochs = 1
num_of_classes = 4251 # len(np.unique(y))# 4251
print('num_of_classes: ', num_of_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=input_shape,padding='same'))
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
model.add(Dense(num_of_classes, activation='softmax'))


model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])
model.summary()

model.fit(np.array(x), train_Y_one_hot, batch_size=batch_size, epochs=num_of_epochs, verbose=1)
