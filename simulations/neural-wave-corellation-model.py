# NEURAL WAVE CORRELATION MODEL
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def neural_wave_model(t, params):
    """
    Modèle unifié onde mécanique / influx nerveux
    Simule les similarités entre propagation d'onde et potentiel d'action
    """
    # Paramètres: amplitude, fréquence, damping, seuil d'activation
    A, f, damping, threshold = params
    
    # Équation d'onde mécanique amortie + caractéristique neurale
    mechanical_wave = A * np.exp(-damping * t) * np.sin(2 * np.pi * f * t)
    neural_signal = 1 / (1 + np.exp(-10 * (mechanical_wave - threshold)))
    
    return mechanical_wave, neural_signal

# Simulation
time = np.linspace(0, 2, 1000)
params = [1.0, 5.0, 2.0, 0.3]  # Amplitude, fréquence, damping, seuil

wave, neural = neural_wave_model(time, params)

# Visualisation comparative
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(time, wave, 'b-', label='Onde mécanique')
plt.title('Propagation ondulatoire')
plt.xlabel('Temps')
plt.ylabel('Amplitude')

plt.subplot(1, 2, 2)
plt.plot(time, neural, 'r-', label='Signal neuronal')
plt.title('Potentiel d action')
plt.xlabel('Temps')
plt.ylabel('Activation')
plt.show()