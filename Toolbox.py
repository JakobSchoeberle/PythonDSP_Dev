import numpy as np
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

def PlotFrequencyDomain(y, N, T):
    yf = fft(y)
    xf = fftfreq(N, T)
    xf = fftshift(xf)
    yplot = fftshift(yf)

    plt.figure
    plt.plot(xf, 1.0/N * np.abs(yplot))
    plt.grid()
    plt.show()

def PlotTimeDomain(t, y):
    plt.figure
    plt.plot(t, y)
    plt.grid()
    plt.show()