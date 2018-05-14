import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
from keras.models import load_model
from keras.utils import to_categorical

from tqdm import tqdm
import numpy as np
import csv

MAX_IMAGES = 200

data = pd.read_csv('data/train.csv')
data['Id_int'] = pd.factorize(data['Id'])[0]

# Maybe not the most elegant solution to find the classes
all_classes = data['Id'].unique()
all_classes_int = data['Id_int'].unique()
print('class: ', all_classes[17], ', Id: ', all_classes_int[17])

test_dir = 'data/test/'
result_file = 'results/sample_submission.csv'
test_paths = pathlib.Path('data/test/').glob('*.jpg')
test_sorted = sorted([x for x in test_paths])
print('Found', len(test_sorted), 'training images')
print(*test_sorted[0:2], sep='\n')

def process(image):
    # resize
    image = np.resize(image, [128, 128, 1])

    # convert to grayscale
    if image.shape == 3:
        image = np.dot([image[:,:,0],image[:,:,1],image[:,:,2]],[0.299,0.587,0.114])

    # return normalized
    return image / 255


def get_image_from_path(path):
    image = mpimg.imread(path)
    image = process(image)

    return image


def loop_over_data(data):
    X = []
    for index, path in tqdm(enumerate(test_sorted)):
        if index > MAX_IMAGES:
            break

        image = get_image_from_path(path)
        X.append(image)

    return np.array(X)


X_test = loop_over_data(test_dir)
model = load_model('whale.h5')
test_eval = model.predict(X_test)

print(test_eval[0].shape)
print(test_eval[0].max())
x = test_eval[0]
max_indices = np.flip(np.argsort(x)[-5:], axis=-1)
max_prob = x[max_indices]
all_classes[max_indices]

with open('results/sample_submission.csv', 'w') as csvfile:
    fieldnames = ['Image', 'Id']
    writer = csv.writer(csvfile)
    #writer.writeheader()
    writer.writerow(fieldnames)
    for index, image_name in enumerate(test_sorted):
        if index > MAX_IMAGES:
            break
        im = str(image_name.parts[-1])
        x = test_eval[index]
        max_indices = np.flip(np.argsort(x)[-5:], axis=-1)
        writer.writerow([im, ' '.join(all_classes[max_indices])])
