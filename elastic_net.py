import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
import io
from PIL import Image


def plank_law(wavelengths, temperature):
	# Planckov zakon zračenja
	h = 6.62607015e-34  # Planckova konstanta [J*s]
	c = 299792458  # brzina svetlosti [m/s]
	k = 1.380649e-23  # Boltzmannova konstanta [J/K]
	
	a = 2 * h * c ** 2
	b = h * c / (wavelengths * k * temperature)
	intensity = a / ((wavelengths ** 5) * (np.exp(b) - 1))
	
	return intensity


if __name__ == '__main__':
 
	# Generisanje skupa trening podataka
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
	
	# Skaliranje podataka
	scaler_X = StandardScaler()
	scaler_Y = StandardScaler()
	
	X_train_scaled = scaler_X.fit_transform(X_train)
	Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1))
	
	# Obučavanje modela ElasticNet regresijom
	model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=1000)
	model.fit(X_train_scaled, Y_train_scaled)
	
	# Generisanje skupa test podataka za predikciju temperature
	for T in [6543, 5400, 3090, 4500, 8000]:
		# fig = plt.figure()

		new_intensity = plank_law(wavelengths, T)
		plt.plot(wavelengths * 1e9, new_intensity, color='red', label='T={}'.format(T))
		
		X_test = new_intensity.reshape(1, -1)
		X_test_scaled = scaler_X.transform(X_test)
	
		
		# Predviđanje temperature
		Y_pred_scaled = model.predict(X_test_scaled)
		y_pred = np.array(Y_pred_scaled).reshape(-1, 1)
		Y_pred = scaler_Y.inverse_transform(y_pred)
		
		predicted_intensity = plank_law(wavelengths, Y_pred[0][0])
		plt.plot(wavelengths * 1e9, predicted_intensity, color='black', label='T={}'.format(Y_pred[0][0]))
		
		# Ispisivanje predviđene temperature
		print(f"Predviđena temperatura za T={T} K je T={Y_pred[0][0]:.2f} K.")


		# plt.savefig(f'spectras/{Y_pred[0][0]:.2f}K.png')

	plt.show()
