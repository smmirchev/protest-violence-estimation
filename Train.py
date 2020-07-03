import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from keras.preprocessing import image
import numpy as np

# Paths to folders with the data
train_path = "D:/01 Capsone Project/ucla-protest/UCLA-protest/img/train"
text_file = "annot_train.txt"
text_file2 = "annot_test.txt"


# convert the txt files to csv files
def convert_to_csv():
    read_file = pd.read_csv(text_file, delim_whitespace=True)
    read_file.to_csv("train_label.csv", index=None)
    read_file = pd.read_csv(text_file2, delim_whitespace=True)
    read_file.to_csv("test_label.csv", index=None)


# convert_to_csv()

col_list = ["fname", "protest", "violence", "sign", "photo", "fire", "police",
            "children", "group_20", "group_100", "flag", "night", "shouting"]

# print(df["protest"])



# change non-numeric data to 0
def convert_non_Numberic():
    # the training data
    df_train = pd.read_csv("train_label.csv", usecols=col_list)
    df_train.loc[(df_train["violence"] == "-") & (df_train["sign"] == "-"), ["violence", "sign"]] = 0, 0
    df_train.loc[(df_train["photo"] == "-"), ["photo"]] = 0
    df_train.loc[(df_train["fire"] == "-") & (df_train["police"] == "-"), ["fire", "police"]] = 0, 0
    df_train.loc[(df_train["children"] == "-") & (df_train["group_20"] == "-"), ["children", "group_20"]] = 0, 0
    df_train.loc[(df_train["group_100"] == "-") & (df_train["flag"] == "-"), ["group_100", "flag"]] = 0, 0
    df_train.loc[(df_train["night"] == "-") & (df_train["shouting"] == "-"), ["night", "shouting"]] = 0, 0
    df_train.to_csv("edit_train_label.csv")

    # the testing data
    df_test = pd.read_csv("test_label.csv", usecols=col_list)
    df_test.loc[(df_test["violence"] == "-") & (df_test["sign"] == "-"), ["violence", "sign"]] = 0, 0
    df_test.loc[(df_test["photo"] == "-"), ["photo"]] = 0
    df_test.loc[(df_test["fire"] == "-") & (df_test["police"] == "-"), ["fire", "police"]] = 0, 0
    df_test.loc[(df_test["children"] == "-") & (df_test["group_20"] == "-"), ["children", "group_20"]] = 0, 0
    df_test.loc[(df_test["group_100"] == "-") & (df_test["flag"] == "-"), ["group_100", "flag"]] = 0, 0
    df_test.loc[(df_test["night"] == "-") & (df_test["shouting"] == "-"), ["night", "shouting"]] = 0, 0
    df_test.to_csv("edit_test_label.csv")


# convert_non_Numberic()
dataset_train = pd.read_csv("edit_train_label.csv")
dataset_test = pd.read_csv("edit_test_label.csv")

# check if dataset contains NA values
# print( dataset_train.isna().sum)

# show data histograms
# sns.pairplot(dataset_train)
# plt.show()

# show data statistics
train_stats = dataset_train.describe()
train_stats.pop("violence")
train_stats = train_stats.transpose()
# print(train_stats)

# remove violence, as it is what needs to be predicted
train_labels = dataset_train.pop("violence")
test_labels = dataset_test.pop("violence")


# create the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', kernel_initializer='normal', input_shape=[len(dataset_train.keys())]),
    layers.Dense(32, activation='relu', kernel_initializer='normal'),
    layers.Dense(1, kernel_initializer='normal', activation="linear")])

optimizer = tf.keras.optimizers.Adam(0.001)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

model.summary()

# example_batch = dataset_train[:10]
# example_result = model.predict(example_batch)
# print(example_result)

history = model.fit( dataset_train, train_labels, epochs=2, batch_size=32,
                     validation_split=0.2, verbose=2)

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
hist.tail()

print("")
loss, mae, mse = model.evaluate(dataset_test, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} violence".format(mae))
print("Testing set Mean Sqe Error: {:5.2f} violence".format(mse))


img = image.load_img('img1.jpg', target_size=(244, 244))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
model.predict(img)

def predict_violence():
    test_predictions = model.predict(dataset_test).flatten()
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values (violence)')
    plt.ylabel('Predictions (violence)')
    points = [0, 1]
    plt.xlim(points)
    plt.ylim(points)
    _ = plt.plot(points, points)
    plt.show()


def plot_MSE():
    plt.plot(history.history['mse'], label='MSE (training data)')
    plt.plot(history.history['val_mse'], label='MSE (validation data)')
    plt.title('MSE for Violence Levels')
    plt.ylabel('MSE value')
    plt.xlabel('No. Epoch')
    plt.legend(loc="upper left")
    plt.show()


def plot_mae():
    plt.plot(history.history['mae'], label="MAE (training data)")
    plt.plot(history.history['val_mae'], label="MAE (validation data)")
    plt.title('MAE for Violence Levels')
    plt.ylabel('MAE Value')
    plt.xlabel('No. Epoch')
    plt.legend(loc='upper left')
    plt.show()


plot_MSE()
plot_mae()
predict_violence()



