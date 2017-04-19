import numpy as np
from keras.applications.vgg19 import VGG19
from keras.optimizers import SGD

TRAIN_SIZE = 10
IMAGE_SIZE = 224

train_images = np.memmap("train_images.npy", dtype='float32', mode='r', shape=(TRAIN_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
train_labels = np.memmap("train_labels.npy", dtype="int32", mode='r+', shape=(TRAIN_SIZE,))

model = VGG19(weights='imagenet', include_top=True)
model.compile(optimizer=SGD(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=128, nb_epoch=30, validation_split=0.2, verbose=1)

json = model.to_json()
with open('model2.json', 'w') as f:
    f.write(json)

model.save_weights("model2_weights.bin")
