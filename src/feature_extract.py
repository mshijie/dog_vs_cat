from keras.applications.inception_v3 import InceptionV3
from keras.layers import *
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import numpy as np

# load data
images = np.load("processed_data/test/images.npy", mmap_mode="r")

# load model
base_model = InceptionV3(weights="imagenet", include_top=False)
outputs = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
feature_extract_model = Model(inputs=base_model.input, outputs=outputs)

# predict
features = feature_extract_model.predict(images, verbose=1)
np.save("processed_data/test/features.npy", features)
