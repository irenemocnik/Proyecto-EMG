# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 13:11:12 2025

@author: iremo
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
import wfdb

print(pywt.wavelist(kind='discrete'))

fs = 4000
def cargaVector (nombre):
    record = wfdb.rdrecord(nombre)
    return record.p_signal.flatten() 

emgHealthy = cargaVector("emg_healthy")
emgNeuro = cargaVector("emg_neuropathy")
emgMyo = cargaVector("emg_myopathy")

wavelet = 'dmey'
niveles = 7
coeffs = pywt.wavedec(emgHealthy, wavelet, level=niveles)
# coeffs = [cA6, cD6, cD5, ..., cD1]

coeffs[0] = np.zeros_like(coeffs[0])      # cA6 aprox hasta 15
coeffs[-1] = np.zeros_like(coeffs[-1])    # cD1 (2000–1000 Hz)
coeffs[-2] = np.zeros_like(coeffs[-2])    # cD2 (1000–500 Hz)

filtrada = pywt.waverec(coeffs, wavelet)
filtrada = filtrada[:len(emgHealthy)] #saca el padding interno


t = np.arange(len(emgHealthy)) / fs


plt.figure(figsize=(10, 4))
plt.plot(t, emgHealthy, label='Original', linewidth=1, alpha=0.4)
plt.plot(t, filtrada, label='Filtrada (15–500 Hz)', linewidth=1)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Filtrado por wavelets (banda 15–500 Hz)')
plt.grid(True)
plt.xlim(3.8, 4.2)
plt.legend()
plt.tight_layout()
plt.show()

