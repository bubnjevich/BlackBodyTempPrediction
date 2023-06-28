import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Conv1D, Dense, Flatten, InputLayer, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import load_model


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
    # Generisanje talasnih dužina u opsegu od 400 nm do 700 nm
    
    # Generisanje temperatura u opsegu od 3000 K do 8000 K
    temperatures = np.linspace(3000, 8000, 100)
    
    # Kreiranje X_train i Y_train
    X_train = []
    Y_train = []
    for temperature in temperatures:
        intensity = plank_law(wavelengths, temperature)
        X_train.append(intensity.flatten())
        Y_train.append([temperature])
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    
    # Skaliranje podataka
    scaler_X.fit(X_train)
    scaler_Y.fit(Y_train)
    X_train_scaled = scaler_X.transform(X_train)
    
    # Inicijalizacija MinMaxScaler
    
    # Skaliranje podataka
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
    # model.save('planck.h5')
    return model


def predikcija(model, X_test_scaled):
    Y_pred_scaled = model.predict(X_test_scaled)
    return Y_pred_scaled


if __name__ == '__main__':
   scaler_X = StandardScaler()
   scaler_Y = StandardScaler()
   model = generate(scaler_X, scaler_Y)

   # izračunajte intenzitete na novim talasnim dužinama
   
   T = [3000, 4000, 6000]
   for t in T:
       new_intensity = plank_law(wavelengths, t)
       plt.plot(wavelengths*1e9, new_intensity, color='red', label='T={}'.format(t))
       
       X_test = [new_intensity.flatten()]
       X_test = np.array(X_test)
       X_test = scaler_X.transform(X_test.reshape(1, -1))
       # izvršite predikciju
       prediction = predikcija(model, X_test)
       
       # transformišite predikciju nazad u originalni oblik
       temperature = scaler_Y.inverse_transform(prediction)[0][0]
       predicted_intensity = plank_law(wavelengths, temperature)
       plt.plot(wavelengths*1e9, predicted_intensity, color='black', label='T={}'.format(temperature))
    
       print(temperature)
       
       
   plt.title("Plankov zakon zračenja")
   plt.xlabel("Talasna dužina [nm]")
   plt.ylabel("Intenzitet [W/(m^2*nm)]")
   x_min, x_max = plt.xlim()
   y_min, y_max = plt.ylim()
   print('Granice x osa: [{}, {}]'.format(x_min, x_max))
   print('Granice y osa: [{}, {}]'.format(y_min, y_max))
   # plt.legend()
   plt.show()

    