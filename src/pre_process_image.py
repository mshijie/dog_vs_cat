import os

import numpy as np
import skimage
import skimage.io
import skimage.transform
import random

IMAGE_SIZE = 224


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


def pre_process_image(folder, output_folder, has_label=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(folder)
    random.shuffle(files)

    length = len(files)
    images = np.zeros((length, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    if has_label:
        labels = np.zeros(length, dtype=np.int32)
    indexes = np.zeros(length, dtype=np.int32)

    for i in range(length):
        file = files[i]
        images[i] = load_image(folder + "/" + file)
        s = file.split(".")
        if has_label:
            labels[i] = 1 if s[0] == 'dog' else 0
            indexes[i] = int(s[1])
        else:
            indexes[i] = int(s[0])

        if i % 100 == 0:
            print("process", folder, i)

    np.save(output_folder + "/images", images)
    if has_label:
        np.save(output_folder + "/labels", labels)
    np.save(output_folder + "/indexes", indexes)


if __name__ == '__main__':
    pre_process_image('data/train', "processed_data/train")
    pre_process_image('data/test', "processed_data/test", has_label=False)
    print("Done.")
