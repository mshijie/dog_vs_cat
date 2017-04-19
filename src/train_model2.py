import numpy as np
from keras.applications.vgg19 import VGG19

train_features = np.load("train_images.npy", mmap_mode='r+')
train_labels = np.load("train_labels.npy", mmap_mode='r+')

model = VGG19(weights='imagenet', include_top=True)

model.fit(train_features, train_labels, batch_size=128, nb_epoch=30, validation_split=0.2, verbose=1)

json = model.to_json()
with open('model2.json', 'w') as f:
    f.write(json)

model.save_weights("model2_weights.bin")
