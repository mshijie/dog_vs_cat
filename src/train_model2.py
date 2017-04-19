import numpy as np
from keras.applications.vgg19 import VGG19
from keras.layers import Dense
from keras.models import Model

TRAIN_SIZE = 25000
IMAGE_SIZE = 224

train_images = np.memmap("train_images.npy", dtype='float32', mode='r', shape=(TRAIN_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
train_labels = np.memmap("train_labels.npy", dtype="int32", mode='r+', shape=(TRAIN_SIZE,))

base_model = VGG19(weights=None, include_top=True)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.get_layer("fc2").output
x = Dense(1024, activation='relu', name="more_fc1")(x)
x = Dense(1024, activation='relu', name="more_fc2")(x)
predictions = Dense(1, activation='sigmoid', name="predictions")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

model.fit(train_images, train_labels, batch_size=64, epochs=30, validation_split=0.2, verbose=1)

json = model.to_json()
with open('model2.json', 'w') as f:
    f.write(json)

model.save_weights("model2_weights.bin")
