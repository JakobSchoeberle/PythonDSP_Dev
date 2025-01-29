# %% ======= Initialization =======
import numpy as np
import math
import sympy
from sympy.solvers import solve
from sympy.abc import x
from scipy import signal
import scipy.io as sio
from scipy.io import wavfile
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

import Toolbox as tools

# %% ======= Importing Data =======

# Wav file import
samplerate, data = wavfile.read("data/music_noise.wav")

# Number of sample points in wav file
N = len(data)

# Sample Spacing
T = 1.0 / 800.0 

# Evenly spaced points from a 0 to 1 used to plot in time domain
t = np.linspace(0, 1, N) 

# Signal data in numpy array
music_noise = np.array(data)


# %% ======= Graphing Data =======

tools.PlotFrequencyDomain(music_noise, N, T)

tools.PlotTimeDomain(t, music_noise)

# %% ======= FIR Filter =======

# Recover the music with a low-pass FIR filter with the following specifications

CutoffFrequency = 4000 # Hz
TransitionBand = 964 # Hz
PassbandRipple = 0.1 # dB
MinStopbandAttenuation = 50 # dB

NormalCutoffFrequency = CutoffFrequency / samplerate # Hz

# %%

Delta1 = solve(PassbandRipple, 20**sympy.log(1+x))
Delta2 = solve(MinStopbandAttenuation, -20**sympy.log(x))

print(Delta1)
print(Delta2)

# %% 

rng = np.random.default_rng()
t = np.linspace(-1, 1, 201)

x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) + 0.1*np.sin(2*np.pi*1.25*t + 1) + 0.18*np.cos(2*np.pi*3.85*t))

xn = x + rng.standard_normal(len(t)) * 0.08

b, a = signal.butter(3, 0.05)

zi = signal.lfilter_zi(b, a)

z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])

z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])\

y = signal.filtfilt(b, a, xn)

plt.figure
plt.plot(t, xn, 'b', alpha=0.75)
plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')
plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice', 'filtfilt'), loc='best')
plt.grid(True)
plt.show()
plt.savefig("output/test.svg")

# %%