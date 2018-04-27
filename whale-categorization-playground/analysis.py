import os.path

import pathlib
import imageio
import numpy as np

import matplotlib.pyplot as plt

training_paths = pathlib.Path('data/train/').glob('*.jpg')
training_sorted = sorted([x for x in training_paths])
print('Found', len(training_sorted), 'training images')
print(*training_sorted[0:2], sep='\n')

# Plot data


def show_images(ims, cmap=None, labels=None):
    plt.figure(figsize=(3 * len(ims), 10))
    if labels is not None:
        assert(len(labels) == len(ims)), 'provide exactly one label per image'
    for idx, im in enumerate(ims):
        plt.subplot(1, len(ims), idx + 1)
        plt.imshow(im, cmap=cmap)
        # plt.axis('off')
        if labels is None:
            plt.title('Image ' + str(idx))
        else:
            plt.title(labels[idx])

    plt.tight_layout()
    plt.show()


ims = list(map(lambda p: imageio.imread(str(p)), training_sorted[0:4]))
show_images(ims)
