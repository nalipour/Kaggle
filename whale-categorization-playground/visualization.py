import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def view_image_index(data, directory, index):
    sampleImageFile = data.Image[index]
    sampleImage = mpimg.imread(directory + sampleImageFile)
    plt.figure()
    plt.imshow(sampleImage)
    plt.show(block=False)


def view_image_whaleID(data, directory, ID):
    indices = data.index[data.Id == ID]
    for index in indices:
        sampleImageFile = data.Image[index]
        sampleImage = mpimg.imread(directory + sampleImageFile)
        plt.figure()
        plt.imshow(sampleImage)
        plt.show(block=False)


def count_individual(data):
    vc = data.Id.value_counts().sort_values(ascending=False)
    vc[:50].plot(kind='bar')
    plt.figure()
    plt.show(block=False)


def plot_accuracy_loss(model):

    print(model.history.keys())
    accuracy = model.history['acc']
    loss = model.history['loss']
    epochs = range(len(accuracy))
    plt.figure()
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=False)
