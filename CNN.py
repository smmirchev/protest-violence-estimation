import keras
from keras import Model, Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Concatenate, UpSampling2D
from pathlib import Path
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import keras as k
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
import pandas as pd
from keras.utils import plot_model
from generator import DataGenerator

# config = tf.ConfigProto(device_count = {'GPU': 1})
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# k.backend.tensorflow_backend.set_session(tf.Session(config=config))

# Path to folder with the data


train_path = "D:/01 Capsone Project/ucla-protest/UCLA-protest/img/train"
batch_size = 128
# create a data generator

# Datasets
partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}


datagen = DataGenerator("D:/01 Capsone Project/ucla-protest/UCLA-protest/img/train",
              "D:/01 Capsone Project/protest model/edit_train_label_orig.csv",
              batch_size)

valid_datagen = DataGenerator("D:/01 Capsone Project/ucla-protest/UCLA-protest/img/test",
              "D:/01 Capsone Project/protest model/edit_test_label_orig.csv",
              batch_size)



# number of files in a directory
def get_total_files(dir_path):
    total_files = 0
    for root, dirs, files in os.walk(dir_path):
        total_files += len(files)
    return total_files


# load and iterate the data sets
# train_batches = datagen.flow_from_directory(train_path, class_mode='categorical', target_size=(128, 128),
#                                             batch_size=batch_size, shuffle=True, subset="training")
# valid_batches = datagen.flow_from_directory(train_path, class_mode='categorical', target_size=(128, 128),
#                                             batch_size=batch_size, shuffle=True, subset="validation")

# confirm the iterator works
# batchX, batchy = train_batches.next()
# print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))


# ///// Build and fine-tune vgg16 model //////

vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
# vgg16_model.summary()

# # transform the model to sequential
# model = Sequential()
# for layer in vgg16_model.layers:
#     model.add(layer)
#
#
# # exclude layers from future training
for layer in vgg16_model.layers:
    layer.trainable = False

dataset_train = pd.read_csv("edit_train_label_orig.csv")
dataset_test = pd.read_csv("edit_test_label_orig.csv")


input = Input(shape=(10, ))

# model = keras.Sequential([
x = Dense(64, activation='relu', kernel_initializer='normal', name="DenseWWWWWW1")(input)
x = Dense(32, activation='relu', kernel_initializer='normal', name="DenseWWWWWWW2")(x)
x = Dense(1, kernel_initializer='normal', activation="linear", name="DenseWWWWWWWW3")(x)

layerX = Flatten(name="DenseWWWWWWWFlatten")(vgg16_model.output)

# Classification
concat_layer = Concatenate(name="DenseWWWWWWWConcat")([x, layerX])
output = Dense(80, name="DenseWWWWWW4", activation='relu')(concat_layer)
output = Dense(10, name="classification", activation='softmax')(output)

# MSE
up_samp = UpSampling2D((2, 2))(vgg16_model.output)
mse_output = Dense(1, kernel_initializer='normal', activation="linear", name="regression")(up_samp)

model = Model(inputs=[vgg16_model.input, input], outputs= [mse_output])

model.summary()

# /////// train the vgg16 model ////////

# parallel_model = multi_gpu_model(model, gpus=2)


model.compile(
              optimizer=Adam(0.001),
             # loss="mean_squared_error" "categorical_crossentropy",
              loss={"regression": "mean_squared_error", "classification": "categorical_crossentropy"},
              metrics={"regression": ["mse"], "classification": ["accuracy"]}
              )

# train_steps = train_batches.samples / batch_size
# valid_steps = valid_batches.samples / batch_size

valid_data = "D:/01 Capsone Project/ucla-protest/UCLA-protest/img/test"
valid_labels = "D:/01 Capsone Project/protest model/edit_test_label_orig.csv"

model.fit_generator(datagen, validation_data=valid_datagen, validation_steps=40, steps_per_epoch=40, epochs=40,
                    verbose=1, workers=6, use_multiprocessing=False)


# save neural network structure
model_structure = model.to_json()
f = Path("model_structure001.json")
f.write_text(model_structure)

# save neural network's trained weights
model.save_weights("model_weights001.h5")

# save the whole model
model.save("whole_model001.h5")
