import os
import pickle

from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D


# create model
base_model = InceptionV3(weights=None, include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
predictions = Dense(1, activation='sigmoid', name="predictions")(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])


# train model
def read_batch(dir, files):
    for batch_file in files:
        with open(dir + "/" + batch_file, "r") as f:
            batch_data = pickle.load(f)
        train_features = batch_data['features']
        train_labels = [1 if name[0] == 'dog' else 0 for name in batch_data['names']]
        yield train_features, train_labels


data_dir = "processed_data/train"
batch_files = os.listdir(data_dir)
x = int(len(batch_files) * 0.8)
train_files = batch_files[0:x]
validation_files = batch_files[x:]

model.fit_generator(read_batch(data_dir, train_files),
                    steps_per_epoch=20000,
                    validation_data=read_batch(data_dir, validation_files),
                    validation_steps=5000,
                    epochs=50, verbose=1)

# save model
os.makedirs("model", exist_ok=True)
json = model.to_json()
with open('model/model.json', 'w') as f:
    f.write(json)
model.save_weights("model/model_weights.bin")
