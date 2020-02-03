import matplotlib.pyplot as plt
import numpy as np

df_cafe = np.array([0.67, 0.69, 0.74, 0.76, 0.78, 0.79, 0.84])
df_kitchen = np.array([0.68, 0.72, 0.74, 0.77, 0.78, 0.79, 0.79])
df = (df_cafe + df_kitchen) / 2.

pv_cafe = np.array([0.88, 0.94, 0.96, 0.96, 0.96, 0.97, 0.98])
pv_kitchen = np.array([0.93, 0.96, 0.97, 0.97, 0.97, 0.98, 0.98])
pv = (pv_cafe + pv_kitchen) / 2.

snr = [6, 9, 12, 15, 18, 21, 24]

plt.plot(snr, df, color='r', marker='x', label='Dialogflow')
plt.plot(snr, pv, color='b', marker='o', label='Picovoice')
plt.xlim(6, 24)
plt.xlabel('SNR dB')
plt.ylim(0.6, 1)
plt.ylabel('Accuracy (Command Acceptance Probability)')
plt.xticks([6, 9, 12, 15, 18, 21, 24])
plt.legend()
plt.grid()
plt.show()
