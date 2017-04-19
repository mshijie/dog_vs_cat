from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD

train_features = np.load("train_features.npy", mmap_mode='r+')
train_labels = np.load("train_labels.npy", mmap_mode='r+')

FEATURE_SIZE = 7 * 7 * 512

input = Input(shape=FEATURE_SIZE, name='input')
net = Dense(1024, activation='relu', name='dense1')(input)
net = Dense(1024, activation='relu', name='dense1')(net)
predict = Dense(1, activation='sigmoid', name='predict')(net)

model = Model(input=input, output=predict)
model.compile(optimizer=SGD(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_features, train_labels, batch_size=128, nb_epoch=30, validation_split=0.2, verbose=1)
