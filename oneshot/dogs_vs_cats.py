"""
We use pre-trained image model to build embeddings with annoy to build approximate nearest neighbours
to perform a prediction
"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import tqdm
from annoy import AnnoyIndex
import glob
import os

model = VGG16(weights='imagenet', include_top=False)

# img_path = 'elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# features = model.predict(x)
batch_size = 16
data_cat = image_dataset_from_directory(directory='dogs-vs-cats/train/cat', image_size=(224, 224), batch_size=batch_size, shuffle=False, labels=None)


counter = 1

for x in tqdm.tqdm(data_cat):
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.reshape(batch_size, -1)
    np.save(f"dogs-vs-cats-numpy/{counter}", features)
    counter += 1
    if counter > 10:
        break

del data_cat
print(f"dog indices start at {counter}")
data_dog = image_dataset_from_directory(directory='dogs-vs-cats/train/dog', image_size=(224, 224), batch_size=batch_size, shuffle=False, labels=None)

for x in tqdm.tqdm(data_dog):
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.reshape(batch_size, -1)
    np.save(f"dogs-vs-cats-numpy/{counter}", features)
    counter += 1
    if counter > 20:
        break

del data_dog

# # load numpy array to memory for building annoy
f = 25088
t = AnnoyIndex(f, 'angular') 
fname = glob.glob("dogs-vs-cats-numpy/*")
indx = 0
cat_indx = []
dog_indx = []
for f in fname:
    item_index = int(os.path.basename(f).split(".")[0])
    v = np.load(f)
    for i in range(v.shape[0]):
        t.add_item(indx, v[i, :])
        indx += 1
        if item_index > 10:
            dog_indx.append(indx)
        else:
            cat_indx.append(indx)

t.build(50)
t.save('dog-vs-cats.ann')


def get_estimate(indxs, label_0, label_1):
    l0 = 0
    l1 = 0
    for indx in indxs:
        if indx in label_0:
            l0 += 1
        else:
            l1 += 1
    print(l0, l1)
    return 0 if l0 > l1 else 1

output = t.get_nns_by_vector(v[1, :], 50)
get_estimate(output, cat_indx, dog_indx)

output = t.get_nns_by_vector(t.get_item_vector(dog_indx[10]), 50)
get_estimate(output, cat_indx, dog_indx)


# now try and predict on test....for fun
batch_size = 128
data_test = image_dataset_from_directory(directory='dogs-vs-cats/test1', image_size=(224, 224), batch_size=batch_size, shuffle=False, labels=None)
pred = []
for x in tqdm.tqdm(data_test):
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.reshape(batch_size, -1)
    for i in range(batch_size):
        output = t.get_nns_by_vector(features[i, :], 50)
        pred.append(get_estimate(output, cat_indx, dog_indx))

output = zip(pred, sorted(glob.glob("dogs-vs-cats/*")))
# output this as predictions
