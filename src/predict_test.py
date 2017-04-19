import numpy as np
from keras.models import model_from_json

# load data
images = np.load("processed_data/test/images.npy", mmap_mode="r")
indexes = np.load("processed_data/test/indexes.npy")

# load model
with open("model/model.json") as f:
    json = "\n".join(f.readlines())
    model = model_from_json(json)
model.load_weights("model/model_weights.bin")

# predict
predict = model.predict(images, verbose=1)
with open("predict.txt", "w") as f:
    f.write("id,label\n")
    for i in range(len(predict)):
        f.write(str(indexes[i]))
        f.write(",")
        f.write(str(predict[i][0]))
        f.write("\n")
