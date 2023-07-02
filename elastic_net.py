import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
import io
from PIL import Image


def plank_law(wavelengths, temperature):

	h = 6.62607015e-34
	c = 299792458
	k = 1.380649e-23
	
	a = 2 * h * c ** 2
	b = h * c / (wavelengths * k * temperature)
	intensity = a / ((wavelengths ** 5) * (np.exp(b) - 1))
	
	return intensity


def predict(file):
	
	plt.clf()
	temperature_str = file.split('/')[-1]
	temperature_str = temperature_str.rstrip('.txt')
	T = float(temperature_str[1:-1])
	

	wavelengths = np.arange(1e-9, 3e-6, 1e-9)
	temperatures = np.linspace(3000, 8000, 100)
	
	X_train = []
	Y_train = []
	for temperature in temperatures:
		intensity = plank_law(wavelengths, temperature)
		X_train.append(intensity)
		Y_train.append(temperature)
	
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	
	scaler_X = StandardScaler()
	scaler_Y = StandardScaler()
	
	X_train_scaled = scaler_X.fit_transform(X_train)
	Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1))
	
	model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=1000)
	model.fit(X_train_scaled, Y_train_scaled)
	
	
	X_test_scaled = np.loadtxt(file)
	X_test_scaled = X_test_scaled.reshape(1, -1)
	
	new_intensity = plank_law(wavelengths, T)
	plt.plot(wavelengths * 1e9, new_intensity, color='red', label='T={}'.format(T))
	
	Y_pred_scaled = model.predict(X_test_scaled)
	y_pred = np.array(Y_pred_scaled).reshape(-1, 1)
	Y_pred = scaler_Y.inverse_transform(y_pred)
	
	predicted_intensity = plank_law(wavelengths, Y_pred[0][0])
	plt.plot(wavelengths * 1e9, predicted_intensity, color='black', label='T={}'.format(Y_pred[0][0]))
	plt.title("Planck's law of radiation")
	plt.xlabel("Wavelength [nm]")
	plt.ylabel("Intensity [W/(m^2*nm)]")
	plt.legend()
	if not os.path.exists('resultsNet'):
		os.makedirs('resultsNet')
	plt.savefig(f'resultsNet/{Y_pred[0][0]:.2f}K.png')

	return Y_pred[0][0], T

