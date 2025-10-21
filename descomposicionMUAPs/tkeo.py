# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:32:33 2025

@author: iremo
"""
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal.windows import dpss



def cargaVector (nombre):
    record = wfdb.rdrecord(nombre)
    return record.p_signal.flatten() 

emgHealthy = cargaVector("emg_healthy")
emgNeuro = cargaVector("emg_neuropathy")
emgMyo = cargaVector("emg_myopathy")


from scipy.signal import butter, filtfilt
fs = 4000
# Filtro pasa banda 20-450 Hz con orden 4




#Cuando hay un muap, el valor de la energia local aumenta bruscamente,
def preac(sig):
    b, a = butter(N=4, Wn=[20, 450], btype='bandpass', fs=fs)
    sig = filtfilt(b, a, sig)
    sigCentrada = sig - np.median(sig)
    mad = np.median(np.abs(sigCentrada))
    sigW = sigCentrada / (1.4826*mad+1e-12)
    return sig, sigCentrada, sigW

Hfilt, Hcent, HAcondicionada = preac(emgHealthy)
Mfilt, Mcent, MAcondicionada = preac(emgMyo)
Nfilt, Ncent, NAcondicionada = preac(emgNeuro)


t = np.arange(0, len(HAcondicionada))/fs


plt.figure(figsize=(10,4))
plt.plot(t, Hfilt, lw=1)
plt.title("Filtrada (20–450 Hz)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud (filtrada)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(4,4.5)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, HAcondicionada, lw=1)
plt.title("Blanqueada (z-score robusto, MAD)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Unidades z")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(4,4.5)
plt.show()

sd = float(np.std(HAcondicionada))
mad = float(np.median(np.abs(HAcondicionada-np.median(HAcondicionada))))



def TKEO(sig):
    psi = sig[1:-1]**2 - sig[:-2]*sig[2:]
    return np.r_[0,psi,0] #agrego cero al principio y al final para mantener la longitud
#y represent energia instantanea, lo usamos de umbral aora detectar picos}

psiSano = TKEO(HAcondicionada)
plt.figure(figsize=(8,6))
plt.plot(psiSano)
plt.ylabel('Energía Instantánea')
plt.grid(True)



    
def detectorpicos(psi, fs, k = 5, refract = 2.5, anchoMin = 0.8):   
    
    
    med = np.median(psi)
    mad = np.median(np.abs(psi - med))
    T = med + k*(mad + 1e-12)
    superior = psi > T
    segmentos = []
    adentro = False
    inicio = None
    for i in range(len(superior)):
        if superior[i] and not adentro:
            adentro = True
            inicio = i
        elif not superior[i] and adentro:
            final = i-1
            segmentos.append((inicio, final))
            adentro = False
            inicio = None
    if adentro:
        segmentos.append((inicio, len(superior)-1)) #caso ultimo segmento termina superando el umbral
    
    
    picos = []
    minVent = int(anchoMin*1e-3*fs) #en unidades d muestras
    for (ini, fin) in segmentos:
        if (fin - ini + 1) < minVent:
            continue #descarta breves
        seg = psi[ini:fin + 1]
        pos_rel = int(np.argmax(seg))
        picos.append(ini + pos_rel)

   
    picos = np.array(sorted(picos), dtype=int)
    refractMuestras = int(refract * 1e-3 * fs)
    validos = []
    ultimo = -10**12
    for p in picos:
        if p - ultimo >= refractMuestras:
            validos.append(p)
            ultimo = p
    picos = np.array(validos, dtype=int)

    return picos, T, segmentos

picosH, umbralH, segmentosH = detectorpicos(psiSano, fs, k = 320, refract = 3.5, anchoMin = 0.8)

t = np.arange(len(psiSano))/fs

            
plt.figure(figsize=(12,4))
plt.plot(t, psiSano, lw=1.0, label="ψ[n] (TKEO)")
plt.hlines(umbralH, t[0], t[-1], linestyles="dashed", label=f"Umbral T = mediana + k·MAD", alpha=0.9)

    # Sombrear segmentos por encima del umbral
for (ini, fin) in segmentosH:
    plt.axvspan(t[ini], t[fin], color="orange", alpha=0.12)

    # Marcar picos en rojo
if len(picosH):
    plt.plot(t[picosH], psiSano[picosH], "ro", ms=5, label=f"Picos ({len(picosH)})")

plt.xlabel("Tiempo [s]")
plt.ylabel("Energía ψ[n]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.xlim(4,4.3)
plt.show()
        


def snippets(sig, picos, fs, pre = 2.0, pos = 4.0):
    prev  = int(round(pre  * 1e-3 * fs))
    post = int(round(pos * 1e-3 * fs))
    L = prev + post + 1
    
    validos = (picos >= prev) & (picos < len(sig)-post)
    picosValidos = picos[validos]
    S = np.stack([sig[p - prev : p + post + 1] for p in picosValidos], axis=0) #eje centrado en el pico
    tVentana = (np.arange(L) - prev) / fs

    return S, picosValidos, prev, post, tVentana

S, picosValidos, prev, post, tVentana = snippets(HAcondicionada, picosH, fs)

M, L = S.shape
plt.figure(figsize=(7,4))
plt.plot(tVentana*1000, S.T, alpha=0.2)
plt.plot(tVentana*1000, S.mean(axis=0), lw=2, label="Promedio")
plt.xlabel("Tiempo relativo al pico [ms]")
plt.ylabel("Amplitud (unidades z)")
plt.title(f"Snippets centrados (M={M}, L={L})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def PCA_DBSCAN(S, componentes=3, eps=0.8, minS=15):
    # PCA clásico: sin random_state ni whiten (compatibilidad total)
    pca = PCA(n_components=componentes)
    X = pca.fit_transform(S)
    
    # Clustering con DBSCAN
    labels = DBSCAN(eps=10, min_samples=10).fit_predict(X)
    return X, pca, labels

X, pca, labels = PCA_DBSCAN(S, componentes=3, eps=0.8, minS=15)

plt.figure(figsize=(6,5))
for lab in np.unique(labels):
    mask = labels == lab
    if lab == -1:
        plt.scatter(X[mask,0], X[mask,1], s=15, alpha=0.3, color='gray', label="Ruido (-1)")
    else:
        plt.scatter(X[mask,0], X[mask,1], s=20, alpha=0.8, label=f"Cluster {lab}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Snippets en espacio PCA (PC1 vs PC2)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()