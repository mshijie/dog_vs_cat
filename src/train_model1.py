import os
import pickle

from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD

BATCH_SIZE = 1000
FEATURE_SIZE = 7 * 7 * 512

# create model
inputs = Input(shape=(FEATURE_SIZE,), name='inputs')
net = Dense(1024, activation='relu', name='dense1')(inputs)
net = Dense(1024, activation='relu', name='dense2')(net)
predict = Dense(1, activation='sigmoid', name='predict')(net)
model = Model(inputs=inputs, outputs=predict)
model.compile(optimizer=SGD(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])


# train model
def read_batch(data_dir, files):
    for batch_file in files:
        with open(data_dir + "/" + batch_file, "r") as f:
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
                    steps_per_epoch=BATCH_SIZE,
                    validation_data=read_batch(data_dir, validation_files),
                    validation_steps=BATCH_SIZE,
                    epochs=50, verbose=1)

# save model
os.makedirs("model", exist_ok=True)
json = model.to_json()
with open('model/model1.json', 'w') as f:
    f.write(json)
model.save_weights("model/model1_weights.bin")
