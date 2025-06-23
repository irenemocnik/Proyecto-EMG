# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 21:37:34 2025

@author: iremo

Las señales son un EMG intramuscular del músculo tibialis anterior.
El procedimiento consiste en la dorsiflexión suave del pie contra resistencia, seguido de relajación.

----------------------------------------------------------------------------------------------------

Datos EMG - Healthy
1 canal,
fs = 4000 Hz,
cantMuestras = 50860 (12 segundos aprox)
El paciente no tiene antecedentes de enfermedades neuromusculares.

----------------------------------------------------------------------------------------------------

Datos EMG - Neuropathy
1 canal, 
fs = 4000 Hz, 
cantMuestras = 147858 (aprox 35 segundos)
El paciente en cuestión tiene neuropatía por radiculopatía L5.

----------------------------------------------------------------------------------------------------

Datos EMG – Myopathy
1 canal, 
fs = 4000 Hz, 
cantMuestras = 110337 (aprox 27 segundos aprox)
El paciente tiene diagnostico de miopatía por una historia de polimiositis prolongada,
esta bajo tratamiento con esteroides y metotrexato en baja dosis.


"""

#%%

import wfdb
import numpy as np
import matplotlib.pyplot as plt

def cargaVector (nombre):
    record = wfdb.rdrecord(nombre)
    return record.p_signal.flatten() 

emgHealthy = cargaVector("emg_healthy")
emgNeuro = cargaVector("emg_neuropathy")
emgMyo = cargaVector("emg_myopathy")





