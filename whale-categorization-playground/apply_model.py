import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
from keras.models import load_model
from keras.utils import to_categorical
import helper

from tqdm import tqdm
import numpy as np
import csv

MAX_IMAGES = 10

data, all_classes = helper.read_classes('data/train.csv')

# Maybe not the most elegant solution to find the classes

test_dir = 'data/test/'
result_file = 'results/sample_submission.csv'
test_paths = pathlib.Path('data/test/').glob('*.jpg')
test_sorted = sorted([x for x in test_paths])
print('Found', len(test_sorted), 'training images')
print(*test_sorted[0:2], sep='\n')

X_test = helper.load_test_data(test_sorted, max_images=MAX_IMAGES)
model = load_model('whale.h5')
test_eval = model.predict(X_test)

with open('results/sample_submission.csv', 'w') as csvfile:
    fieldnames = ['Image', 'Id']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for index, image_name in tqdm(enumerate(test_sorted)):
        if index > MAX_IMAGES:
            break
        im = str(image_name.parts[-1])
        x = test_eval[index]
        max_indices = np.flip(np.argsort(x)[-5:], axis=-1)
        writer.writerow([im, ' '.join(all_classes[max_indices])])
