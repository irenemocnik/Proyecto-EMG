# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:17:40 2025

@author: iremo
"""


import numpy as np
import pywt
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import resample_poly
#print(pywt.wavelist(kind='discrete'))

fs = 4000
def cargaVector (nombre):
    record = wfdb.rdrecord(nombre)
    return record.p_signal.flatten() 

emgHealthy = cargaVector("emg_healthy")
emgNeuro = cargaVector("emg_neuropathy")
emgMyo = cargaVector("emg_myopathy")

# # largoDeseado = len(emgHealthy) # 50860
# # largoNeuro = len(emgNeuro) # 147858
# # largoMyo = len(emgMyo) # 110337

# #para llevar neuro de 147858 a 50860 muestras, interpolamos por 50860 y diezmamos por 147858 (73929/25430)
# emgNeuro = resample_poly(emgNeuro, up=25430, down=73929)
# emgMyo   = resample_poly(emgMyo, up=50860, down=110337)



wavelet = 'dmey'
niveles = 7

def filtroW(senal):
    coeffs = pywt.wavedec(senal, wavelet, level=niveles)
# coeffs = [cA6, cD6, cD5, ..., cD1]

    coeffs[0] = np.zeros_like(coeffs[0])      # cA6 aprox hasta 15
    coeffs[-1] = np.zeros_like(coeffs[-1])    # cD1 (2000–1000 Hz)
    coeffs[-2] = np.zeros_like(coeffs[-2])    # cD2 (1000–500 Hz)

    filtrada = pywt.waverec(coeffs, wavelet)
    filtrada = filtrada[:len(senal)] #saca el padding interno
    return filtrada

emgHealthyF = filtroW(emgHealthy)
emgMyoF = filtroW(emgMyo)
emgNeuroF = filtroW(emgNeuro)

def rectificar(senal):
    senalRectificada = np.abs(senal - np.mean(senal))
    return senalRectificada
rectH = rectificar(emgHealthyF)
rectN = rectificar(emgNeuroF)
rectM = rectificar(emgMyoF)

def filtroPromediador(senal, fs, largoVentana):
    N = int(fs * largoVentana / 1000) #tamaño en unidad de muestras
    kernel = np.ones(N) / N
    return np.convolve(senal, kernel, mode='same')


from scipy.signal import butter, filtfilt

nyq = fs / 2
orden = 4
fc = [5, 10, 20, 40]
envolventesLP = {}
for f in fc:
    b, a = butter(orden, f / nyq, btype='low')
    envolvente = filtfilt(b, a, rectH)
    envolventesLP[f] = envolvente
    


def envRMS(senal, fs, largoVentana):
    N = int(fs * largoVentana / 1000) 
    senal = senal**2 #Energía
    kernel = np.ones(N) / N
    rms = np.convolve(senal, kernel, mode = 'same') #Convolución entre una ventana rectangular y la senal
    return np.sqrt(rms)



envolventePromH = filtroPromediador(rectH, fs = fs, largoVentana = 50)
envolventePromN = filtroPromediador(rectN, fs = fs, largoVentana = 50)
envolventePromM = filtroPromediador(rectM, fs = fs, largoVentana = 50)

envolventeRMSH = envRMS(emgHealthyF, fs = fs, largoVentana = 80)
envolventeRMSN = envRMS(emgNeuroF, fs = fs, largoVentana = 80)
envolventeRMSM = envRMS(emgMyoF, fs = fs, largoVentana = 80)

tN = np.arange(len(emgNeuro)) / fs
tM = np.arange(len(emgMyo)) / fs
tH = np.arange(len(emgHealthy)) / fs
plt.figure(figsize=(12, 4))

t = tH
# # Envolvente ARV  movil
# plt.figure()
# plt.plot(t, rectH, label='Paciente saludable', alpha=0.8)
# plt.plot(t, envolventePromH, label='Envolvente (Promediador)', color='blue', linewidth=1)
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Amplitud [mV]")
# plt.title("Envolvente de EMG")
# plt.legend()
# plt.tight_layout()
# plt.show()


# # Envolventes LP a distintas fc
# plt.figure()
# plt.plot(t, rectH, label='Paciente saludable', alpha=0.8, color = 'steelblue')
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Amplitud [mV]")
# plt.title("Envolvente LP")
# for f in fc:
#     plt.plot(t,envolventesLP[f], label=f'fc = {f} Hz')
# plt.legend()
# plt.tight_layout()
# plt.xlim(3.0,5.0)
# plt.show()

plt.figure(figsize=(12, 4))
plt.plot(tH, rectH, label='Rectificación de onda completa',color='steelblue', alpha=0.5)
plt.plot(tH, envolventeRMSH, label='Envolvente (RMS)', color='steelblue', linewidth=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.title("Visualización temporal - Paciente saludable")
plt.legend()
plt.tight_layout()
plt.ylim(0,1)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(tN, rectN, label='Rectificación de onda completa',color='green', alpha=0.5)
plt.plot(tN, envolventeRMSN, label='Envolvente (RMS)',color='green')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.title("Visualización temporal - Paciente con neuropatía")
plt.legend()
plt.ylim(0,4)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(tM, rectM, label='Rectificación de onda completa',color='red', alpha=0.5)
plt.plot(tM,  envolventeRMSM, label='Envolvente (RMS)', color='red', linewidth=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.title("Visualización temporal - Paciente con miopatía")
plt.legend()
plt.ylim(0,1)
plt.tight_layout()
plt.show()



