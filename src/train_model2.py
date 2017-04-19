import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD

TRAIN_SIZE = 25000
IMAGE_SIZE = 224

train_images = np.memmap("train_images.npy", dtype='float32', mode='r', shape=(TRAIN_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
train_labels = np.memmap("train_labels.npy", dtype="int32", mode='r+', shape=(TRAIN_SIZE,))

base_model = InceptionV3(weights="imagenet", include_top=False)


# for layer in base_model.layers:
#     layer.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', name="more_fc1")(x)
x = Dense(1024, activation='relu', name="more_fc2")(x)
predictions = Dense(1, activation='sigmoid', name="predictions")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])


model.fit(train_images, train_labels, batch_size=128, epochs=100, validation_split=0.2, verbose=1)

json = model.to_json()
with open('model2.json', 'w') as f:
    f.write(json)

model.save_weights("model2_weights.bin")

print("model saved")
