import os

from keras.applications.inception_v3 import InceptionV3
from keras.layers import *
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD

# create model
base_model = InceptionV3(weights="imagenet", include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
predictions = Dense(1, activation='sigmoid', name="predictions")(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])

# train model
images = np.load("processed_data/train/images.npy", mmap_mode="r")
labels = np.load("processed_data/train/labels.npy")
model.fit(images, labels, batch_size=32, epochs=10, verbose=1, validation_split=0.2)

# save model
os.makedirs("model", exist_ok=True)
json = model.to_json()
with open('model/model.json', 'w') as f:
    f.write(json)
model.save_weights("model/model_weights.bin")
