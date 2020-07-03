
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications import vgg16
import csv

# load json and create model
json_file = open('model_final_structure04.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_final_weights04.h5")
csv_path ="D:/01 Capsone Project/protest model/edit_train_label_orig.csv"

model.summary()

imagePath = "train-00083.jpg" # the image path
num_vector = [1,0,0,0,0,1,0,0,0,0] # the categories present

def getImage(img_data):
    # load an image file to test, resizing it to 64x64 pixels (as required by this model)
    img = image.load_img(img_data, target_size=(128, 128))

    # convert the image to a numpy array
    image_array = image.img_to_array(img)

    # add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
    single_image = np.expand_dims(image_array, axis=0)

    # normalize the data
    single_image = vgg16.preprocess_input(single_image)

    return single_image


def classifyImage():
    image = getImage(imagePath)
    #vector_1 = ["sign", "photo", "fire", "police", "children", "group_20", "group_100", "flag", "night", "shouting"]


    with open(csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

# predict has a slight bug when as it expects a batch
# to fix it the below lines have to be used which will add and extra dimension to aux
    aux = num_vector
    aux = np.asarray(aux)
    print(aux.shape)
    aux=aux.reshape((1,10))
    print(aux.shape)
    aux2 = [image]

    results = model.predict([image, aux])
    print(results)



classifyImage()

