import os

import numpy as np
import skimage
import skimage.io
import skimage.transform
import pickle
import random
from keras.applications.vgg16 import VGG16
from keras.models import Model

IMAGE_SIZE = 224
BATCH_SIZE = 1000


base_vgg_model = VGG16(weights='imagenet', include_top=True)
feature_extract_model = Model(inputs=base_vgg_model.input, outputs=base_vgg_model.get_layer('fc1').output)


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
    names = []
    for i, file in enumerate(files):
        images[i] = load_image(path + "/" + file)
        names.append(file.split(".")[0:-1])
    return images, names


def get_batches(files):
    length = len(files)
    total_batch = (length + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(total_batch):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        if end > length:
            end = length
        yield i, files[start: end]


def pre_process_image(folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    all_files = os.listdir(folder)
    random.shuffle(all_files)

    for i, files in get_batches(all_files):
        images, names = load_all_image(folder, files)
        features = feature_extract_model.predict(images)

        batch_data = {"images": images, "names": names, "features": features}
        with open(output_folder + "/batch_" + str(i), "wb") as f:
            pickle.dump(batch_data, f)
            print("process batch", folder, i)


if __name__ == '__main__':
    pre_process_image('data/train', "processed_data/train")
    pre_process_image('data/test', "processed_data/test")
    print("Done.")
