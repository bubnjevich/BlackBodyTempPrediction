import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Conv1D, Dense, Flatten, InputLayer, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import pickle

h = 6.626e-34
c = 3.0e+8
k = 1.38e-23

def plank_law(wav, T):
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity


wavelengths = np.arange(1e-9, 3e-6, 1e-9)


def generate(scaler_X, scaler_Y):
    
    temperatures = np.linspace(3000, 8000, 100)
    
    X_train = []
    Y_train = []
    for temperature in temperatures:
        intensity = plank_law(wavelengths, temperature)
        X_train.append(intensity.flatten())
        Y_train.append([temperature])
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    
    scaler_X.fit(X_train)
    scaler_Y.fit(Y_train)
    X_train_scaled = scaler_X.transform(X_train)
    
    
    Y_train_scaled = scaler_Y.transform(Y_train)
    
    return CNN(X_train_scaled, Y_train_scaled)

def CNN(X_train_scaled, Y_train_scaled):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train_scaled, Y_train_scaled, epochs=50, batch_size=16)
    model.save(f'cnnModel/planck.h5')
    
    return model



def cnn_prediction(model_file, file):
    
    model = load_model(model_file)
    
    temperature_str = file.split('/')[-1]
    temperature_str = temperature_str.rstrip('.txt')
    T = float(temperature_str[1:-1])
    
    X_test_scaled = np.loadtxt(file)
    X_test_scaled = X_test_scaled.reshape(1, -1)
    
    prediction = model.predict(X_test_scaled)
    
    new_intensity = plank_law(wavelengths, T)
    plt.clf()
    plt.plot(wavelengths*1e9, new_intensity, color='red', label='T={}'.format(T))
    
    with open(f'scalers/scaler_Y.pkl', 'rb') as f:
        scalerY = pickle.load(f)

        temperature = scalerY.inverse_transform(prediction)[0][0]
        predicted_intensity = plank_law(wavelengths, temperature)
        plt.plot(wavelengths*1e9, predicted_intensity, color='black', label='T={}'.format(temperature))
    
        plt.title("Planck's law of radiation")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Intensity [W/(m^2*nm)]")
        plt.legend()
        plt.savefig(f'resultsCNN/{temperature:.2f}K.png')
        f.close()
        return temperature, T


if __name__ == '__main__':
   scaler_X = StandardScaler()
   scaler_Y = StandardScaler()
   model = generate(scaler_X, scaler_Y)
   
   with open(f'scalers/scaler_X.pkl', 'wb') as f:
       pickle.dump(scaler_X, f)
   
   with open(f'scalers/scaler_Y.pkl', 'wb') as f:
       pickle.dump(scaler_Y, f)

    