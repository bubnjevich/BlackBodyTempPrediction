import matplotlib.pyplot as plt

from elastic_net import *


def test():
    wavelengths = np.arange(1e-9, 3e-6, 1e-9)
    new_intensity = plank_law(wavelengths, 3840)
    plt.plot(wavelengths * 1e9, new_intensity, color='red', label='T={}'.format(3840))
    plt.savefig(f'test/{3840:.2f}K.png')
    plt.clf()

if __name__ == '__main__':

    wavelengths = np.arange(1e-9, 3e-6, 1e-9)
    T = 5321
    new_intensity = plank_law(wavelengths, T)
    plt.plot(wavelengths * 1e9, new_intensity, color='red', label='T={}'.format(T))
    plt.savefig(f'test/{T:.2f}K.png')
    plt.clf()
    exit(0)



    temperatures = np.linspace(5593, 8000, 500)
    X_train = []
    Y_train = []
    for T in temperatures:
        new_intensity = plank_law(wavelengths, T)
        plt.plot(wavelengths * 1e9, new_intensity, color='red', label='T={}'.format(T))
        plt.savefig(f'spectras/{T:.2f}K.png')
        plt.clf()

