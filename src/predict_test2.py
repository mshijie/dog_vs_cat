import numpy as np
from keras.models import model_from_json

test_features = np.load("test_images.npy", mmap_mode='r+')

with open("model2.json") as f:
    json = "\n".join(f.readlines())
    model = model_from_json(json)

model.load_weights("model2_weights.bin")

predict = model.predict(test_features, verbose=1)

with open("predict2.txt", "w") as f:
    f.write("id,label\n")
    for i in range(len(predict)):
        f.write(str(i+1))
        f.write(",")
        f.write(str(predict[i][0]))
        f.write("\n")
