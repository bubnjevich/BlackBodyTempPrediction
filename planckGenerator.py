
from elastic_net import *


def generate_X_test_scaled(T):
	
	wavelengths = np.arange(1e-9, 3e-6, 1e-9)
	temperatures = np.linspace(3000, 8000, 100)
	
	X_train = []
	Y_train = []
	for temperature in temperatures:
		intensity = plank_law(wavelengths, temperature)
		X_train.append(intensity)
		Y_train.append(temperature)
	
	X_train = np.array(X_train)
	
	scaler_X = StandardScaler()
	scaler_X.fit_transform(X_train)

	wavelengths = np.arange(1e-9, 3e-6, 1e-9)
	new_intensity = plank_law(wavelengths, T)
	X_test = new_intensity.reshape(1, -1)
	X_test_scaled = scaler_X.transform(X_test)


	temperature_str = str(T)
	filename = f"spectrasNet/T{temperature_str}K.txt"
	np.savetxt(filename, X_test_scaled)

	return X_test_scaled

if __name__ == '__main__':
	for T in [2890, 4321, 2332]:
		generate_X_test_scaled(T)