import os

import numpy as np
import skimage
import skimage.io
import skimage.transform

IMAGE_SIZE = 224
BATCH_SIZE = 100


def load_image(file):
    img = skimage.io.imread(file)
    img = img / 255
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resize_img = skimage.transform.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE, 3), mode='reflect')
    return resize_img.astype(np.float32)


def load_all_image(path, files):
    length = len(files)
    images = np.zeros((length, IMAGE_SIZE, IMAGE_SIZE, 3))
    labels = np.zeros(length)
    for i, file in enumerate(files):
        images[i] = load_image(path + "/" + file)
        labels[i] = 1 if "dog" in file else 0
    return images, labels


def get_batches(files):
    length = len(files)
    total_batch = (length + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(total_batch):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        if end > length:
            end = length
        yield start, end, files[start: end]


def pre_process_image(folder, images_file, label_file):
    files = os.listdir(folder)

    all_images = np.memmap(images_file, dtype='float32', mode='w+', shape=(len(files), IMAGE_SIZE, IMAGE_SIZE, 3))
    if label_file:
        all_labels = np.memmap(label_file, dtype='int32', mode='w+', shape=(len(files),))

    for start, end, files in get_batches(files):
        images, labels = load_all_image(folder, files)
        all_images[start:end] = images
        if label_file:
            all_labels[start:end] = labels
        print("process image", folder, start, end)

    all_images.flush()
    if label_file:
        all_labels.flush()


if __name__ == '__main__':
    pre_process_image('data/train', "train_images.npy", "train_labels.npy")
    pre_process_image('data/test', "test_image.npy", None)
    print("data saved")
