import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
import pathlib

def process(image):
    # resize
    image = np.resize(image, [128, 128, 1])

    # convert to grayscale
    if image.shape == 3:
        image = np.dot([image[:, :, 0], image[:, :, 1],
                        image[:, :, 2]], [0.299, 0.587, 0.114])

    # return normalized
    return image / 255


def get_image_from_path(image_name, df=None, with_labels=True):
    image = mpimg.imread(image_name)
    image = process(image)
    if(with_labels):
        label = df[df.Image == pathlib.Path(image_name).parts[-1]]['Id_int'].values[0]
        return image, label
    else:
        return image, None


def loop_over_data(df, data_dir, max_images=None):
    X = []
    y = []
    for index, image_name in tqdm(enumerate(df.Image)):
        if max_images and index > max_images:
            break
        image, label = get_image_from_path(
            data_dir+image_name, df, True)
        X.append(image)
        y.append(label)

    return np.array(X), y


def loop_over_test_data(data, max_images=None):
    X = []
    for index, image_name in tqdm(enumerate(data)):
        if max_images and index > max_images:
            break

        image = get_image_from_path(image_name, with_labels=False)
        X.append(image[0])

    return np.array(X)
