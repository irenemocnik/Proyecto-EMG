# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 10:57:14 2025

@author: iremo
"""
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import find_peaks
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



def envRMS(senal, fs, largoVentana):
    N = int(fs * largoVentana / 1000) 
    senal = senal**2 #Energía
    kernel = np.ones(N) / N
    rms = np.convolve(senal, kernel, mode = 'same') #Convolución entre una ventana rectangular y la senal
    return np.sqrt(rms)

envolventeRMSH = envRMS(emgH, fs = fs, largoVentana = 500)
envolventeRMSN = envRMS(emgN, fs = fs, largoVentana = 500)
envolventeRMSM = envRMS(emgM, fs = fs, largoVentana = 500)

tN = np.arange(len(emgNeuro)) / fs
tH = np.arange(len(emgHealthy)) / fs
tM = np.arange(len(emgM)) / fs

resultados = []

def metricas(envolventeTotal, tiempoTotal, envolvente, tiempo, nombre):
   
    mean = np.mean(envolvente)
    pico = np.max(envolvente)
    area = np.trapz(envolvente, dx = 1/fs )
    duracion= len(envolvente) / fs
    duracionRelativa = len(envolvente) / len(envolventeTotal) 
    
    resultados.append({
        "Segmento": nombre,
        "Media [V]": round(mean, 4),
        "Pico [V]": round(pico, 4),
        "Área [Vs]": round(area, 4),
        "Duración [s]": round(duracion, 4),
        "Duración [%]": round(duracionRelativa * 100, 2),
    })
    


#regs de interes SALUDABLE: 
actBajaH = envolventeRMSH[0 : int(3.3 * fs)]
actFuerteH = envolventeRMSH[int(3.3 * fs) : int(4.45 * fs)]   
relajacionH = envolventeRMSH[int(4.45 * fs): int(len(envolventeRMSH)*fs)]                             

t1H = tH[0 : int(3.3 * fs)]
t2H = tH[int(3.3 * fs) : int(4.45 * fs)]   
t3H = tH[int(4.45 * fs): int(len(envolventeRMSH)*fs)]                                     

metricas(envolventeRMSH, tH, actBajaH, t1H, "Saludable - Actividad Moderada")
metricas(envolventeRMSH, tH, actFuerteH, t2H, "Saludable - Actividad Fuerte")
metricas(envolventeRMSH, tH, relajacionH, t3H, "Saludable - Relajación")


#regs de interes NEUROPATÍA: 
actBajaN = envolventeRMSN[0 : int(11.5 * fs)]
actFuerteN = envolventeRMSN[int(11.5 * fs) : int(18.8 * fs)]   
relajacionN = envolventeRMSN[int(18.8 * fs): int(len(envolventeRMSN)*fs)]                             

t1N = tN[0 : int(11.5 * fs)]
t2N = tN[int(11.5 * fs) : int(18.8 * fs)]   
t3N = tN[int(18.8 * fs): int(len(envolventeRMSN)*fs)]                                     

metricas(envolventeRMSN,tN,actBajaN, t1N, "Neuropatía - Actividad Moderada")
metricas(envolventeRMSN,tN,actFuerteN, t2N, "Neuropatía - Actividad Fuerte")
metricas(envolventeRMSN,tN,relajacionN, t3N, "Neuropatía - Relajación")

# regs de interes MIO

actBajaM = envolventeRMSM[0 : int(12.5 * fs)]
actFuerteM = envolventeRMSM[int(12.5 * fs) : int(15 * fs)]   
relajacionM = envolventeRMSM[int(15 * fs): int(len(envolventeRMSM)*fs)]                             

t1M = tM[0 : int(12.5 * fs)]
t2M = tM[int(12.5 * fs) : int(15 * fs)]   
t3M = tM[int(15 * fs): int(len(envolventeRMSM)*fs)]                                     

metricas(envolventeRMSM,tM,actBajaM, t1M, "Miopatía - Actividad Moderada")
metricas(envolventeRMSM,tM,actFuerteM, t2M, "Miopatía - Actividad Fuerte")
metricas(envolventeRMSM,tM,relajacionM, t3M, "Miopatía - Relajación")

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
plt.savefig("metricas_emg.png", bbox_inches='tight', dpi=300)
plt.show()
