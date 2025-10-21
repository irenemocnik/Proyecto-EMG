# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 16:39:31 2025

@author: iremo
"""

import numpy as np
import matplotlib.pyplot as plt
import wfdb
import scipy.signal as sig
import pandas as pd
from pandas.plotting import table

fs = 4000
def cargaVector (nombre):
    record = wfdb.rdrecord(nombre)
    return record.p_signal.flatten() 

emgHealthy = cargaVector("emg_healthy")
emgNeuro = cargaVector("emg_neuropathy")
emgMyo = cargaVector("emg_myopathy")

    
    

from scipy.signal import butter, filtfilt

# BUTTER pasa banda 20-450 Hz con orden 4
b, a = butter(N=4, Wn=[15, 450], btype='bandpass', fs=fs)
emgButter = filtfilt(b, a, emgMyo)
emgH = filtfilt(b, a, emgHealthy)
emgN = filtfilt(b, a, emgNeuro)
recorte1 = emgButter[0:int(16.75*fs)]
recorte2 = emgButter[int(17*fs):int(26.5*fs)]

# Concatenarlos
emgM = np.concatenate((recorte1, recorte2))


def psdWelch (senal):
    N = len(senal)
    nperseg_altares = N // 4
    noverlap_altares = nperseg_altares // 2
    #baja resolucion menos varianza
    nperseg_bajavar = N // 50 #muchos segmentos muy cortos
    noverlap_bajavar = int(nperseg_bajavar * 0.7) #MUCHO solapamiento
    
    f1, Pxx1 = sig.welch(senal, fs=fs, window='hamming',
                     nperseg=nperseg_altares, noverlap=noverlap_altares,
                     detrend='linear', scaling='density')
    area1 = np.cumsum(Pxx1)

    f2, Pxx2 = sig.welch(senal, fs=fs, window='hamming',
                     nperseg=nperseg_bajavar, noverlap=noverlap_bajavar,
                     detrend='linear', scaling='density')

    area2 = np.cumsum(Pxx2)

    return f1,Pxx1, f2, Pxx2, area1, area2

  

fH, PSDH, fH2, PSDH2, AH1, AH2 = psdWelch(emgH)
fN, PSDN, fN2, PSDN2, AN1, AN2 = psdWelch(emgN)
fM, PSDM, fM2, PSDM2, AM1, AM2 = psdWelch(emgM)



  

#gráficos de PSD
plt.figure(figsize=(10, 5))
plt.plot(fH, 10 * np.log10(PSDH), label=f'Alta resolución ', color='steelblue')
plt.plot(fH2, 10 * np.log10(PSDH2), label=f'Muy baja varianza', color='blue')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia [dB]')
plt.title('PSD Sano - Comparación: resolución vs varianza')
plt.legend()
plt.xlim([0, 500])
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(fN, 10 * np.log10(PSDN), label=f'Alta resolución ', color='green', alpha = 0.5)
plt.plot(fN2, 10 * np.log10(PSDN2), label=f'Muy baja varianza', color='green')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia [dB]')
plt.title('PSD Neuropatía - Comparación: resolución vs varianza')
plt.legend()
plt.xlim([0, 500])
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(fM, 10 * np.log10(PSDM), label=f'Alta resolución ', color='red', alpha = 0.5)
plt.plot(fM2, 10 * np.log10(PSDM2), label=f'Muy baja varianza', color='red')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad espectral de potencia [dB]')
plt.title('PSD Miopatía - Comparación: resolución vs varianza')
plt.legend()
plt.xlim([0, 500])
plt.grid(True)
plt.tight_layout()

resultados = []

def metricas(psd, f, nombre):
    pico = f[np.argmax(psd)]
    valorMean = np.sum(f*psd) / np.sum(psd)
    areaTotal = np.trapz(psd, f)
    resultados.append({
        "Segmento": nombre,
        "MPF [Hz]": round(valorMean, 4),
        "Pico [Hz]": round(pico, 4),
        "Área [V²]": round(areaTotal, 4),
    })
    
metricas(PSDH,fH, "Paciente Saludable")
metricas(PSDM,fM, "Paciente Miopatía")
metricas(PSDN,fN, "Paciente Neuropatía")

df = pd.DataFrame(resultados)
print(df)

fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 1))  # altura dinámica
ax.axis('off')
colWidths = [0.35] + [0.12] * (len(df.columns) - 1)
tabla = table(ax, df, loc='center', cellLoc='center', colWidths=colWidths)
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 1.5)

# Guardar imagen
plt.savefig("metricasPSD_emg.png", bbox_inches='tight', dpi=300)
plt.show()


