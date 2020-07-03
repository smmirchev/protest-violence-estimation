import numpy as np
import keras
import csv
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16

class DataGenerator(keras.utils.Sequence):
    train_path = "D:/01 Capsone Project/ucla-protest/UCLA-protest/img/train"
    csv_path ="D:/01 Capsone Project/protest model/edit_train_label_orig.csv"
    'Generates data for Keras'
    def __init__(self, train_path, csv_path, batch_size):
        'Initialization'
        self.train_path = train_path
        self.csv_path = csv_path
        self.batch_size = batch_size

        self.list_csv = []
        with open(self.csv_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for First, fname, protest, violence, sign, photo, fire, police, children, group_20, group_100, flag, night, shouting in spamreader:
                vector_1 = [sign, photo, fire, police, children, group_20, group_100, flag, night, shouting]
                vector_2 = violence
                vector_3 = self.train_path + "/" + fname
                self.list_csv.append([vector_1, vector_2, vector_3])

        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_csv) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_csv[k] for k in indexes]

        # Generate data
        X1, X2, y, = self.__data_generation(list_IDs_temp)

        # return X, y
        return [np.asarray(X1, dtype='float32'), np.asarray(X2, dtype='float32')], np.asarray(y, dtype='float32')

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_csv))
        np.random.shuffle(self.indexes)

    def getImage(self, img_data):
        # load an image file to test, resizing it to 64x64 pixels (as required by this model)
        img = image.load_img(img_data, target_size=(128, 128))

        # convert the image to a numpy array
        image_array = image.img_to_array(img)

        # add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
        single_image = np.expand_dims(image_array, axis=0)

        # normalize the data
        single_image = vgg16.preprocess_input(single_image)

        return single_image[0]


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = []
        X2 = []
        y = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # print(ID)
            # print(type(X2))

            X2.append(ID[0])
            # Store sample
            X1.append(self.getImage(ID[2]))

            # load image

            # Store class
            y.append(ID[1])

        return X1, X2, y
#         return [np.asarray(X1, dtype='float32'), np.asarray(X2, dtype='float32')], np.asarray(y, dtype='float32')

