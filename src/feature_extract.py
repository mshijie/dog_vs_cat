import os

import numpy as np
import skimage
import skimage.io
import skimage.transform
from keras.applications.vgg19 import VGG19
from keras.models import Model

IMAGE_SIZE = 224
FEATURE_SIZE = 7 * 7 * 512
BATCH_SIZE = 20

base_vgg_model = VGG19(weights='imagenet', include_top=True)
feature_extract_model = Model(inputs=base_vgg_model.input, outputs=base_vgg_model.get_layer('flatten').output)


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


def extract_features(folder, max_size=-1):
    files = os.listdir(folder)[0:max_size]

    features = np.zeros((len(files), FEATURE_SIZE))
    all_labels = np.zeros(len(files))

    for start, end, files in get_batches(files):
        images, labels = load_all_image(folder, files)
        print("load image", folder, start, end)
        predict = feature_extract_model.predict(images)
        features[start:end] = predict
        all_labels[start:end] = labels
        print("extract feature", folder, start, end)
    return features, all_labels


if __name__ == '__main__':
    train_features, train_labels = extract_features('data/train')
    test_features, _ = extract_features('data/test')
    np.save("train_features.npy", train_features)
    np.save("train_labels.npy", train_labels)
    np.save("test_features.npy", test_features)
    print("data saved")
