import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Conv1D, Dense, Flatten, InputLayer, MaxPooling1D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img
import cv2
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from PIL import Image
import os

h = 6.626e-34
c = 3.0e+8
k = 1.38e-23


def plank_law(wav, T):
    a = 2.0 * h * c ** 2
    b = h * c / (wav * k * T)
    intensity = a / ((wav ** 5) * (np.exp(b) - 1.0))
    return intensity


wavelengths = np.arange(1e-9, 3e-6, 1e-9)


def generate(scaler_X, scaler_Y):
    img_size = (200, 120)
    X_train = []
    for file_name in os.listdir("spectras/"):
        img_path = os.path.join("spectras/", file_name)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, img_size)
        X_train.append(img_resized)

    X_train = np.array(X_train)

    Y_train = []
    for file_name in os.listdir("spectras/"):
        temp_str = file_name.split('K')[0]
        temp = float(temp_str)
        Y_train.append(temp)

    Y_train = np.array(Y_train).reshape(-1, 1)

    return CNN(X_train, Y_train)

def CNN(X, Y):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.fit(X_train, Y_train, epochs=50, batch_size=32)
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(loss)
    print(accuracy)
    return model





def predikcija(model, X_test_scaled):
    Y_pred_scaled = model.predict(X_test_scaled)
    return Y_pred_scaled


if __name__ == '__main__':
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    model = generate(scaler_X, scaler_Y)

    # izračunajte intenzitete na novim talasnim dužinama

    img_size = (640, 480)

    img = load_img("test/5321.00K.png", target_size=img_size)
    img_array = img_to_array(img)
    X_train = img_array.reshape((1,) + img_array.shape)
    print(predikcija(model, X_train))


